import math
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import numpy as np
import os
from functools import partial
from src.model.train_util.logger import CSVLogger
from src.loss.seg_loss import get_loss_fn as get_seg_loss_fn
from src.loss.seg_loss import get_recon_loss_follow_seg
from src.loss.seg_loss import get_dice_score, get_seg_precision, get_seg_recall
from src.loss.class_loss import get_loss_fn as get_class_loss_fn
from src.loss.class_loss import get_class_accuracy, get_class_precison, get_class_recall

class UnexpectedBehaviorException(Exception):
    def __init__(self, message="An unintended exception occurred."):
        super().__init__(message)

# refer site: https://www.deepspeed.ai/docs/config-json/
def get_deepspeed_config_dict(train_dataset, loader_batch_size, batch_size, num_workers,
                              stage_coef_list=[10, 90], decay_epoch=100,
                              cycle_min_lr=4e-5, cycle_max_lr=2e-4, decay_lr_rate=0.25):
    train_dataloader_len = math.ceil(len(train_dataset) / loader_batch_size)
    num_gpu = torch.cuda.device_count()
    train_batch_size = batch_size
    micro_batch_size = loader_batch_size // num_gpu
    accumulation_steps = train_batch_size // loader_batch_size
    epoch_step_size = math.ceil(train_dataloader_len // accumulation_steps)
    ds_config = {
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        #"gradient_accumulation_steps": 2,
        "gradient_clipping": 1.0,
        "pipeline_parallel": True,
        "steps_per_print": epoch_step_size,
        ## 정밀도는 bf16에 비해 높으나, gradient overflow, underflow 문제가 빈번하여 bf16사용 추천
        "fp16": {
            "enabled": False,
            "loss_scale": 0, # 0 means dynamic loss_scale, another value means fixed loss scaling
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "consecutive_hysteresis": False,
            # 문서에는 int를 넣으라고 되어있지만, float을 넣어도 동작한다.
            # 하지만 print때 약간의 이슈 발생
            "min_loss_scale": 1, 
        },
        "bf16": {
        "enabled": False 
        },
        "zero_optimization": {
            "stage": 2,  # ZeRO-2를 활성화
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
            ###############################
            "offload_optimizer": {
                "device": "cpu", # [cpu|nvme]
                "nvme_path": "/local_nvme",
                "pin_memory": True,
                "buffer_count": 4,
                "fast_init": False
            },
            "offload_param": {
                "device": "cpu", # [cpu|nvme]
                "nvme_path": "/local_nvme",
                "pin_memory": True,
                "buffer_count": 5,
                "buffer_size": 1e8,
                "max_in_cpu": 1e9
            }
        },
    #     "scheduler": {
    #         "type": "WarmupLR",
    #         "params": {
    #             "warmup_min_lr": 2e-6,
    #             "warmup_max_lr": 2e-5,
    #             "warmup_num_steps": 1000,
    #             "warmup_type": "log"
    #         }
    #     },
    "data_sampling": {
        "enabled": True,
        "num_workers": num_workers,
    },
        "scheduler": {
            "type": "OneCycle",
            "params": {
                "cycle_min_lr": cycle_min_lr,
                "cycle_max_lr": cycle_max_lr,
                "decay_lr_rate": decay_lr_rate,
                "cycle_first_step_size": epoch_step_size * stage_coef_list[0],
                "cycle_second_step_size": epoch_step_size * stage_coef_list[1],
                "decay_step_size": epoch_step_size * decay_epoch,
            }
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-5,
                "betas": [0.9, 0.999]
            }
        }
    }
    return ds_config

def toggle_grad(target_model, require_grads=False):
    for param in target_model.parameters():
        param.requires_grad = require_grads

def average_across_gpus(torch_scalar):
    dist.barrier()
    dist.reduce(torch_scalar, dst=0, op=dist.ReduceOp.AVG)
    return torch_scalar.item()

def gather_across_gpus(torch_scalar, is_scalar=True):
    dist.barrier()
    gathered_scalar_list = [torch.zeros_like(torch_scalar) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_scalar_list, torch_scalar)
    if is_scalar:
        concatenated_scalar_list = [gathered_scalar.item() for gathered_scalar in gathered_scalar_list]
    else:
        concatenated_scalar_list = [gathered_scalar.cpu() for gathered_scalar in gathered_scalar_list]
    return concatenated_scalar_list

def load_deepspeed_model_to_torch_model(torch_model, weight_path):
    # map_location cpu is for reduction gpu memory usage
    state_dict = torch.load(weight_path, map_location=torch.device("cpu"))
    state_dict = {key.replace("module.", ""): value
                for key, value in state_dict.items()}
    torch_model.load_state_dict(state_dict)

def get_loss_fn_dict(get_seg, get_class, get_recon, use_seg_in_recon,
                     seg_loss_str="propotional_bce_focal", class_loss_str="bce"):
    loss_fn_dict = {}
    if get_seg:
        loss_fn_dict["get_seg_loss"] = get_seg_loss_fn(seg_loss_str, per_image=True)
    if get_class:
        loss_fn_dict["get_class_loss"] = get_class_loss_fn(class_loss_str, per_image=True)
    if get_recon:
        if use_seg_in_recon:
            get_recon_loss = partial(get_recon_loss_follow_seg, per_image=True)
        else:
            get_l1_loss = nn.L1Loss(reduction='none')
            get_recon_loss = lambda y_recon_pred, y_recon_gt, _: get_l1_loss(y_recon_pred, y_recon_gt)        
        loss_fn_dict["get_recon_loss"] = get_recon_loss
    return loss_fn_dict

def get_metric_fn_dict(get_seg, get_class):
    metric_fn_dict = {}
    if get_seg:
        metric_fn_dict["get_dice_score"] = partial(get_dice_score, per_image=True)
        metric_fn_dict["get_seg_precision"] = partial(get_seg_precision, per_image=True)
        metric_fn_dict["get_seg_recall"] = partial(get_seg_recall, per_image=True)
    if get_class:
        metric_fn_dict["get_class_accuracy"] = partial(get_class_accuracy, mask_threshold=None, per_channel=True, exclude_class_0=True)
        metric_fn_dict["get_class_precision"] = partial(get_class_precison, mask_threshold=None, per_channel=True, exclude_class_0=True)
        metric_fn_dict["get_class_recall"] = partial(get_class_recall, mask_threshold=None, per_channel=True, exclude_class_0=True)

def get_loss_score_dict(seg_num_classes):
    return {
        "loss_list": [],
        "dice_score_list": [[] for _ in range(seg_num_classes - 1)],
        "seg_precision_list": [[] for _ in range(seg_num_classes - 1)],
        "seg_recall_list": [[] for _ in range(seg_num_classes - 1)],
        "class_y_pred_gt_pair_list": [],
        "recon_diff_list": []
    }

def update_loss_score_dict(loss_score_dict, get_seg, get_class, get_recon, metric_dict):
    if metric_dict is not None:
        loss_score_dict["loss_list"].extend(metric_dict["loss"])
        if get_seg:
            loss_score_dict["dice_score_list"].extend(metric_dict["dice_score"])
            loss_score_dict["seg_precision_list"].extend(metric_dict["seg_precision"])
            loss_score_dict["seg_recall_list"].extend(metric_dict["seg_recall"])
        if get_class:
            loss_score_dict["class_y_pred_gt_pair_list"].extend(metric_dict["class_y_pred_gt_pair"])
        if get_recon:
            loss_score_dict["recon_diff_list"].extend(metric_dict["recon_diff"])

def convert_log_scalar_to_str(log_scalar, precision=3):
    format_str = f'{{:.{precision}f}}'
    log_str = format_str.format(log_scalar)
    return log_str

def get_log_path(base_folder, batch_size, get_seg, get_class, get_recon, use_seg_in_recon, prefix=""):
    if dist.get_rank() == 0:
        log_path = base_folder
        if prefix != "":
            log_path = f"{log_path}/{prefix}_{batch_size}"
        else:
            log_path = f"{log_path}/{batch_size}"
        if get_seg:
            log_path = f"{log_path}_seg"
        if get_class:
            log_path = f"{log_path}_class"
        if get_recon:
            if use_seg_in_recon:
                log_path = f"{log_path}_recon_with_seg"
            else:
                log_path = f"{log_path}_recon"
        os.makedirs(f"{log_path}/weights", exist_ok=True)
        os.makedirs(f"{log_path}/plots", exist_ok=True)
        return log_path
    else:
        return None
    
def get_csv_logger(log_path, get_seg, get_class, get_recon):
    if dist.get_rank() == 0:
        epoch_col = ["epoch"]
        train_col = ["loss"]
        val_col = ["val_loss"]
        if get_seg:
            train_col.append("dice_score")
            val_col.append("val_dice_score")
        if get_class:
            train_col.append("accuracy")
            val_col.append("val_accuracy")
        if get_recon:
            train_col.append("max_recon_diff")
            val_col.append("val_max_recon_diff")
        csv_logger = CSVLogger(f"{log_path}/log.csv", epoch_col + train_col + val_col)
        return csv_logger
    else:
        return None
    
def print_and_save_log(train_loss_score_dict, val_loss_score_dict, csv_logger, epoch,
                       get_seg, get_class, get_recon, metric_fn_dict, precision=3):
    if dist.get_rank() == 0:
        data_info_str = [f'{epoch}']
        
        get_log_str_with_round = partial(convert_log_scalar_to_str, precision=precision)

        train_loss = get_log_str_with_round(np.mean(train_loss_score_dict["loss_list"]))
        val_loss = get_log_str_with_round(np.mean(val_loss_score_dict["loss_list"]))
        train_info_str = [train_loss]
        val_info_str = [val_loss]
        if get_seg:
            train_dice_score = get_log_str_with_round(np.mean(train_loss_score_dict["dice_score_list"]))
            train_seg_precision = get_log_str_with_round(np.mean(train_loss_score_dict["seg_precision_list"]))
            train_seg_recall = get_log_str_with_round(np.mean(train_loss_score_dict["seg_recall_list"]))
            
            val_dice_score = get_log_str_with_round(np.mean(val_loss_score_dict["dice_score_list"]))
            val_seg_precision = get_log_str_with_round(np.mean(val_loss_score_dict["seg_precision_list"]))
            val_seg_recall = get_log_str_with_round(np.mean(val_loss_score_dict["seg_recall_list"]))

            train_info_str.extend([train_dice_score, train_seg_precision, train_seg_recall])
            val_info_str.extend([val_dice_score, val_seg_precision, val_seg_recall])
        if get_class:
            def get_log_str_list_from_vector(vector):
                return [get_log_str_with_round(scalar.item()) for scalar in vector]
            
            get_class_accuracy_fn = metric_fn_dict["get_class_accuracy"]
            get_class_precison_fn = metric_fn_dict["get_class_precison"]
            get_class_recall_fn = metric_fn_dict["get_class_recall"]

            train_class_y_pred_gt_pair_list = train_loss_score_dict["class_y_pred_gt_pair_list"]
            train_class_y_pred_list, train_class_y_true_list = zip(*train_class_y_pred_gt_pair_list)
            val_class_y_pred_gt_pair_list = val_loss_score_dict["class_y_pred_gt_pair_list"]
            val_class_y_pred_list, val_class_y_true_list = zip(*val_class_y_pred_gt_pair_list)

            train_class_y_pred_list = torch.cat(train_class_y_pred_list, dim=0)
            train_class_y_true_list = torch.cat(train_class_y_true_list, dim=0)
            val_class_y_pred_list = torch.cat(val_class_y_pred_list, dim=0)
            val_class_y_true_list = torch.cat(val_class_y_true_list, dim=0)

            train_class_accuracy_vector = get_class_accuracy_fn(train_class_y_pred_list, train_class_y_true_list)
            train_class_precision_vector = get_class_precison_fn(train_class_y_pred_list, train_class_y_true_list)
            train_class_recall_vector = get_class_recall_fn(train_class_y_pred_list, train_class_y_true_list)
            val_class_accuracy_vector = get_class_accuracy_fn(val_class_y_pred_list, val_class_y_true_list)
            val_class_precision_vector = get_class_precison_fn(val_class_y_pred_list, val_class_y_true_list)
            val_class_recall_vector = get_class_recall_fn(val_class_y_pred_list, val_class_y_true_list)

            train_class_accuracy_list = get_log_str_list_from_vector(train_class_accuracy_vector)
            train_class_precision_list = get_log_str_list_from_vector(train_class_precision_vector)
            train_class_recall_list = get_log_str_list_from_vector(train_class_recall_vector)
            
            val_class_accuracy_list = get_log_str_list_from_vector(val_class_accuracy_vector)
            val_class_precision_list = get_log_str_list_from_vector(val_class_precision_vector)
            val_class_recall_list = get_log_str_list_from_vector(val_class_recall_vector)

            train_info_str.extend(train_class_accuracy_list)
            train_info_str.extend(train_class_precision_list)
            train_info_str.extend(train_class_recall_list)

            val_info_str.extend(val_class_accuracy_list)
            val_info_str.extend(val_class_precision_list)
            val_info_str.extend(val_class_recall_list)

        if get_recon:
            train_recon_diff = get_log_str_with_round(train_loss_score_dict["recon_diff_list"])
            val_recon_diff = get_log_str_with_round(val_loss_score_dict["recon_diff_list"])

            train_info_str.append(train_recon_diff)
            val_info_str.append(val_recon_diff)

        data_info_str = data_info_str + train_info_str + val_info_str
        csv_logger.writerow([*data_info_str])
        print_info_str = " - ".join(data_info_str)
        print(print_info_str)

def seg_onehot_and_permute(idx_tensor, num_classes):
    img_dim = idx_tensor.dim() - 1
    permute_tuple = (0, -1, *range(1, 1 + img_dim))
    onehot_tensor = F.one_hot(idx_tensor, num_classes=num_classes)
    onehot_tensor = onehot_tensor.permute(permute_tuple)
    return onehot_tensor

def seg_idx_mask_to_class_label(idx_tensor, num_classes):
    img_dim = idx_tensor.dim() - 1
    sum_dim_tuple = tuple(range(1, 1 + img_dim))
    class_idx_tensor = (idx_tensor.sum(dim=sum_dim_tuple) > 0).long()
    class_label_tensor = F.one_hot(class_idx_tensor, num_classes=num_classes)
    return class_label_tensor

def class_idx_to_class_vector(idx_tensor, num_classes):
    class_label_tensor = F.one_hot(idx_tensor, num_classes=num_classes)
    return class_label_tensor

def get_processed_data(x, y, num_classes, device, dtype, get_seg, get_class, get_recon):
    if get_seg:
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)
        y_label = seg_idx_mask_to_class_label(y, num_classes).to(dtype=dtype)
        y = seg_onehot_and_permute(y, num_classes).to(dtype=dtype)
        return x, y, y_label
    else:
        if get_class:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            y_label = class_idx_to_class_vector(y, num_classes).to(dtype=dtype)
            return x, y_label
        else:
            if get_recon:
                x = x.to(device=device, dtype=dtype)
                return x
            else:
                error_message_str = "You need to set at least one task: get_seg, get_class, or get_recon."
                raise UnexpectedBehaviorException(error_message_str)

def compute_loss_metric(model, process_data_list,
                        num_classes, get_seg, get_class, get_recon,
                        loss_score_dict, loss_fn_dict, metric_fn_dict):
    x, y, y_label = None, None, None
    y_seg_pred, y_label_pred, y_recon_pred = None, None, None

     ######## Unpack Data ########
    if get_seg:
        x, y, y_label = process_data_list
    else:
        if get_class:
            x, y_label = process_data_list
        else:
            x = process_data_list
    ######## Compute Loss ########
    model_output_dict = model(x)
    loss = 0
    if get_seg:
        y_seg_pred = model_output_dict["seg"]
        seg_loss = loss_fn_dict["get_seg_loss"](y_seg_pred, y)
        loss = loss + seg_loss
    if get_class:
        y_label_pred = model_output_dict["class"]
        class_loss = loss_fn_dict["get_class_loss"](y_label_pred, y_label)
        loss = loss + class_loss
    if get_recon:
        y_recon_pred = model_output_dict["recon"]
        recon_loss = loss_fn_dict["get_recon_loss"](y_recon_pred, x, y_seg_pred)
        loss = loss + recon_loss
    ######## Compute Metric #######
    with torch.no_grad():
        if get_seg:
            _, y_seg_pred = torch.max(y_seg_pred, dim=1)
            y_seg_pred = seg_onehot_and_permute(y_seg_pred, num_classes)
            dice_score = metric_fn_dict["get_dice_score"](y_seg_pred, y)
            seg_precision = metric_fn_dict["get_seg_precision"](y_seg_pred, y)
            seg_recall = metric_fn_dict["get_seg_recall"](y_seg_pred, y)
        if get_recon:
            pass
        metric_dict = {
            "loss": gather_across_gpus(loss),
        }
        if get_seg:
            metric_dict["dice_score"] = gather_across_gpus(dice_score)
            metric_dict["seg_precision"] = gather_across_gpus(seg_precision)
            metric_dict["dice_score"] = gather_across_gpus(seg_recall)
        if get_class:
            y_label_pred = gather_across_gpus(y_label_pred, is_scalar=False)
            y_label = gather_across_gpus(y_label, is_scalar=False)
            class_y_pred_gt_pair = [(y_label_pred_item, y_label_item) for y_label_pred_item, y_label_item in zip(y_label_pred, y_label)]
            loss_score_dict["class_y_pred_gt_pair_list"] = class_y_pred_gt_pair
        if get_recon:
            metric_dict["recon_diff"] = gather_across_gpus(recon_loss)
    return loss, metric_dict
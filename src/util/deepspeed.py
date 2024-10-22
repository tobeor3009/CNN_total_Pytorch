import math
import torch
from torch.nn import functional as F
import torch.distributed as dist
import numpy as np
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


def load_deepspeed_model_to_torch_model(torch_model, weight_path):
    # map_location cpu is for reduction gpu memory usage
    state_dict = torch.load(weight_path, map_location=torch.device("cpu"))
    state_dict = {key.replace("module.", ""): value
                for key, value in state_dict.items()}
    torch_model.load_state_dict(state_dict)

def get_loss_score_dict():
    return {
        "loss_list": [],
        "dice_score_list": [],
        "accuracy_list": [],
        "recon_diff_list": []
    }

def update_loss_score_dict(loss_score_dict, get_class, get_recon, metric_dict):
    if metric_dict is not None:
        loss_score_dict["loss_list"].append(metric_dict["loss"])
        loss_score_dict["dice_score_list"].append(metric_dict["dice_score"])
        if get_class:
            loss_score_dict["accuracy_list"].append(metric_dict["accuracy"])
        if get_recon:
            loss_score_dict["recon_diff_list"].append(metric_dict["recon_diff"])

def print_and_save_log(train_loss_score_dict, val_loss_score_dict, csv_logger, epoch, get_class, get_recon):
    if dist.get_rank() == 0:
        data_info_str = [f'{epoch}']
        train_info_str = [f'{np.mean(train_loss_score_dict["loss_list"]):.4f}',
                        f'{np.mean(train_loss_score_dict["dice_score_list"]):.4f}']
        val_info_str = [f'{np.mean(val_loss_score_dict["loss_list"]):.4f}',
                        f'{np.mean(val_loss_score_dict["dice_score_list"]):.4f}']

        if get_class:
            train_info_str.append(f'{np.mean(train_loss_score_dict["accuracy_list"]):.4f}')
            val_info_str.append(f'{np.mean(val_loss_score_dict["accuracy_list"]):.4f}')
        if get_recon:
            train_info_str.append(f'{np.mean(train_loss_score_dict["recon_diff_list"]):.4f}')
            val_info_str.append(f'{np.mean(val_loss_score_dict["recon_diff_list"]):.4f}')

        data_info_str = data_info_str + train_info_str + val_info_str
        csv_logger.writerow([*data_info_str])
        data_info_str = " - ".join(data_info_str)
        print(data_info_str)

def onehot_and_permute(idx_tensor, num_classes):
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

def get_processed_data(x, y, num_classes, device, dtype):
    x = x.to(device=device, dtype=dtype)
    y = y.to(device=device, dtype=torch.long)
    y_label = seg_idx_mask_to_class_label(y, num_classes).to(dtype=dtype)
    y = onehot_and_permute(y, num_classes).to(dtype=dtype)
    return x, y, y_label

def compute_loss_metric(model, x, y, y_label,
                        get_seg_loss, get_class_loss, get_recon_loss,
                        get_dice_score, accuracy_metric,
                        num_classes, get_class, get_recon):
    ######## Compute Loss ########
    model_output = model(x)
    if get_class and get_recon:
        y_pred, y_label_pred, y_recon_pred = model_output
        seg_loss = get_seg_loss(y_pred, y)
        class_loss = get_class_loss(y_label_pred, y_label)
        recon_loss = get_recon_loss(y_recon_pred, x, y_pred)
        loss = seg_loss + class_loss + recon_loss
    elif get_class:
        y_pred, y_label_pred = model_output
        class_loss = get_class_loss(y_label_pred, y_label)
        seg_loss = get_seg_loss(y_pred, y)
        loss = seg_loss + class_loss
    elif get_recon:
        y_pred, y_recon_pred = model_output
        seg_loss = get_seg_loss(y_pred, y)
        recon_loss = get_recon_loss(y_recon_pred, x, y_pred)
        loss = seg_loss + recon_loss
    else:
        y_pred = model_output
        seg_loss = get_seg_loss(y_pred, y)
        loss = seg_loss
    ######## Compute Metric #######
    with torch.no_grad():
        _, y_pred = torch.max(y_pred, dim=1)
        y_pred = onehot_and_permute(y_pred, num_classes)
        dice_score = get_dice_score(y_pred, y)
        accuracy = accuracy_metric(y_label_pred, y_label)
        metric_dict = {
            "loss": average_across_gpus(loss),
            "dice_score": average_across_gpus(dice_score)
        }
        if get_class:
            metric_dict["accuracy"] = average_across_gpus(accuracy)
        if get_recon:
            metric_dict["recon_diff"] = average_across_gpus(recon_loss)
    return loss, metric_dict
import math
import torch
import torch.distributed as dist

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



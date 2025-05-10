import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch import optim
from src.model.train_util.scheduler import OneCycleLR

class UnexpectedBehaviorException(Exception):
    def __init__(self, message="An unintended exception occurred."):
        super().__init__(message)

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

def setup(rank, world_size, master_addr='localhost', master_port='12355'):
    """환경 설정 함수"""
    os.environ['MASTER_ADDR'] = master_addr  # 마스터 노드의 IP 주소 설정
    os.environ['MASTER_PORT'] = master_port     # 마스터 노드의 포트 번호 설정
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def get_model(model_fn, model_config_dict):
    return model_fn(**model_config_dict)

def cleanup():
    """프로세스 그룹 종료 함수"""
    dist.destroy_process_group()

def train(rank, world_size, train_dataset, val_dataset, data_loader_config,
          model_fn, model_config_dict, optimizer_config_dict, loss_fn):
    """분산 학습 함수"""
    setup(rank, world_size)

    data_num_put_each_proc = data_loader_config["data_num_put_each_proc"]
    num_workers_each_proc = data_loader_config["num_workers_each_proc"]
    collate_fn = data_loader_config["collate_fn"]
    total_batch_size = world_size * data_num_put_each_proc

    data_loader_kwarg_dict = {
        "batch_size": data_num_put_each_proc,
        "num_workers": num_workers_each_proc,
        "pin_memory": True,
        "collate_fn": collate_fn
    }

    init_lr = optimizer_config_dict["init_lr"]
    total_epoch = optimizer_config_dict["total_epoch"]
    stage_epoch_list = optimizer_config_dict["stage_coef_list"]
    max_min_lr_ratio = optimizer_config_dict["max_min_lr_ratio"]
    decay_lr = optimizer_config_dict["decay_lr"]

    decay_dropout_ratio = 0.25 ** (1 / total_epoch)
    
    assert len(stage_epoch_list) == 2, f"stage_epoch_list len must be 2"
    # 모델 초기화 및 DDP 설정
    model = get_model(model_fn, model_config_dict).to(rank)
    model = DDP(model, device_ids=[rank])

    # 데이터셋 및 데이터 로더 설정
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, **data_loader_kwarg_dict)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, **data_loader_kwarg_dict)
    train_step_size = len(train_dataloader)
    # 옵티마이저 및 손실 함수 설정
    optimizer = optim.Adam(model.parameters(), lr=init_lr, betas=(0.5, 0.999))
    # StepLR 스케줄러 정의
    scheduler_params = {
        "step_size": train_step_size,
        "first_epoch": stage_epoch_list[0],
        "second_epoch": stage_epoch_list[1],
        "total_epoch": 50,
        "max_min_lr_ratio": max_min_lr_ratio,
        "decay_lr": decay_lr
    }
    scheduler = OneCycleLR(optimizer, **scheduler_params)

    # 학습 루프
    for epoch in range(1, 1 + total_epoch):
        train_sampler.set_epoch(epoch)  # 데이터 셔플을 위한 epoch 설정
        if rank == 0:
            loss_list = []
        else:
            loss_list = None
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(rank), target.to(rank)

            # 옵티마이저 및 손실 계산
            optimizer.zero_grad()
            output = model(data)
            # loss 는 [B] shape 을 기대한다.
            loss_scalar_each = loss_fn(output, target)
            loss = torch.mean(loss_scalar_each)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if rank == 0:
                loss_list_per_gpu = gather_across_gpus(loss_scalar_each)
                loss_list.extend(loss_list_per_gpu)
            
            if batch_idx % 100 == 0:
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}")

    cleanup()


def main():
    """메인 함수"""
    world_size = torch.cuda.device_count()  # 사용 가능한 GPU 수
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
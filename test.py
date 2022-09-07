import os
from src.model.inception_resnet_v2.classification import InceptionResNetV2Transformer3D
import torch
from torch import nn
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from src.data_loader.classification import ClassifyDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from glob import glob
from datetime import date

gpu_number = "0, 1, 2, 3"


def windowing_brain(img_array, channel=3, return_uint8=True):
    img_array = img_array.transpose((2, 0, 1))
#     slice_range = np.arange(img_array.shape[0])
#     slice_range = np.random.choice(slice_range, 32)
#     slice_range = np.sort(slice_range)
#     img_array = img_array[slice_range]
    slice_range = np.arange(img_array.shape[0] - 31)
    slice_range = np.random.choice(slice_range)
    img_array = img_array[slice_range:slice_range + 32]
    if channel == 1:
        img_array = img_array + 40
        img_array = np.clip(img_array, 0, 160)
        img_array = img_array / 160

    elif channel == 3:
        dcm0 = img_array - 5
        dcm0 = np.clip(dcm0, 0, 50)
        dcm0 = dcm0 / 50.

        dcm1 = img_array + 0
        dcm1 = np.clip(dcm1, 0, 80)
        dcm1 = dcm1 / 80.

        dcm2 = img_array + 20
        dcm2 = np.clip(dcm2, 0, 200)
        dcm2 = dcm2 / 200.

        img_array = np.stack([dcm0, dcm1, dcm2], 0)

    if return_uint8:
        return np.uint8(img_array * (2 ** 8 - 1))

    else:  # the value is normalized to [0, 1]
        return img_array


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'
    os.environ['MASTER_PORT'] = '18000'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def label_policy(label):
    age, gender = label.split("_")
    age, gender = float(age), float(gender)
    return np.array([age, gender])


def loss_fun(logit, y, age_loss_fun, gender_loss_fun):
    logit_age = logit[..., 0]
    y_age = y[..., 0]
    logit_gender = logit[..., 1]
    y_gender = y[..., 1].float()
    age_loss = age_loss_fun(logit_age, y_age)
    gender_loss = gender_loss_fun(logit_gender, y_gender)

    return age_loss * 0.9 + gender_loss * 0.1


def metric(logit, y):
    logit_age = logit[..., 0]
    y_age = y[..., 0]
    logit_gender = (logit[..., 1] >= 0.5).float()
    y_gender = y[..., 1].float()
    age_precision = torch.abs(logit_age - y_age).mean()
    gender_acc = 1 - torch.abs(logit_gender - y_gender).mean()

    return age_precision, gender_acc


def main_worker(rank, ngpus_per_node):
    print(f"rank_{rank}")
    torch.cuda.set_device(rank)

    batch_size = 8
    on_memory = False
    augmentation_proba = 0.8
    augmentation_policy_dict = {
        "positional": True,
        "noise": True,
        "elastic": False,
        "brightness_contrast": False,
        "color": False,
        "to_jpeg": False
    }
    image_channel_dict = {"image": "rgb"}
    preprocess_input = windowing_brain
    target_size = (512, 512)
    interpolation = "bilinear"
    class_mode = "binary"
    # class_mode = "categorical"
    dtype = torch.float32

    data_common_path = "./datasets/1.normal_npy/"

    train_image_path_list = glob(f"{data_common_path}/train/*/*.npy")
    valid_image_path_list = glob(f"{data_common_path}/valid/*/*.npy")

    label_list = os.listdir(f"{data_common_path}/train/")

    label_to_index_dict = {label: index for index,
                           label in enumerate(label_list)}
    index_to_label_dict = {index: label for index,
                           label in enumerate(label_list)}

    common_arg_dict = {
        "label_policy": label_policy,
        "augmentation_policy_dict": augmentation_policy_dict,
        "image_channel_dict": image_channel_dict,
        "preprocess_input": preprocess_input,
        "target_size": target_size,
        "interpolation": interpolation,
        "class_mode": class_mode,
        "dtype": dtype
    }

    num_workers = min(batch_size // 2, 8)

    batch_size = int(batch_size / ngpus_per_node)
    num_workers = int(num_workers / ngpus_per_node)

    train_dataset = ClassifyDataset(image_path_list=train_image_path_list,
                                    on_memory=on_memory,
                                    augmentation_proba=augmentation_proba,
                                    **common_arg_dict
                                    )
    valid_dataset = ClassifyDataset(image_path_list=valid_image_path_list,
                                    on_memory=on_memory,
                                    augmentation_proba=0,
                                    **common_arg_dict
                                    )

    setup(rank, ngpus_per_node)

    model = InceptionResNetV2Transformer3D(n_input_channels=3, block_size=8,
                                           padding='valid', z_channel_preserve=True,
                                           dropout_proba=0, num_class=2,
                                           include_context=True, use_base=False).to(rank)

    model = DDP(model, device_ids=[rank])
    train_sampler = DistributedSampler(train_dataset)
    valid_sampler = DistributedSampler(valid_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        #         sampler=train_sampler
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        #         sampler=valid_sampler
    )

    today = date.today()
    today_str = today.strftime("%Y-%m-%d")
    ckpt_dir = '/ckpts'
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)

    start_epoch = 0
    lr = 1e-4
    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    age_loss_fun = nn.L1Loss().to(rank)
    gender_loss_fun = nn.BCELoss().to(rank)

    num_epoch = 100
    check_batch_term = 50

    num_train = len(train_dataset)
    num_train_for_epoch = np.ceil(num_train / batch_size)
    num_val = len(valid_dataset)
    num_val_for_epoch = np.ceil(num_val / batch_size)

    writer_train = SummaryWriter(
        log_dir=f"{log_dir}/{today_str}/gpu_{gpu_number}/train")
    writer_val = SummaryWriter(
        log_dir=f"{log_dir}/{today_str}/gpu_{gpu_number}/val")

    for epoch in range(start_epoch + 1, num_epoch + 1):

        model.train()
        loss_arr = []
        age_precision_arr = []
        gender_acc_arr = []
        for batch, (x, y) in enumerate(train_loader, 1):
            x = x.to(rank)
            y = y.to(rank)

            pred = model(x)

            optim.zero_grad()

            loss_value = loss_fun(pred, y, age_loss_fun, gender_loss_fun)
            age_precision, gender_acc = metric(pred, y)
            loss_arr += [loss_value.item()]
            age_precision_arr += [age_precision.item()]
            gender_acc_arr += [gender_acc.item()]

            loss_value.backward()

            optim.step()
            if batch % check_batch_term == 0:
                print('train : epoch %04d / %04d | Batch %04d \ %04d | Loss %04f | Age Precision %04f | Gender Acc %04f'
                      % (epoch, num_epoch, batch, num_train_for_epoch, np.mean(loss_arr), np.mean(age_precision_arr), np.mean(gender_acc_arr)))
                age_precision_arr = []
                gender_acc_arr = []

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        # validation
        with torch.no_grad():
            model.eval()
            loss_arr = []
            age_precision_arr = []
            gender_acc_arr = []
            for batch, (x, y) in enumerate(valid_loader, 1):
                x = x.to(rank)
                y = x.to(rank)

                pred = model(x)

                loss_value = loss_fun(pred, y)
                age_precision, gender_acc = metric(pred, y)
                loss_arr += [loss_value.item()]
                if batch % check_batch_term == 0:
                    print('valid : epoch %04d / %04d | Batch %04d \ %04d | Loss %04f | Age Precision %04f | Gender Acc %04f'
                          % (epoch, num_epoch, batch, num_val_for_epoch, np.mean(loss_arr), np.mean(age_precision_arr), np.mean(gender_acc_arr)))
                    age_precision_arr = []
                    gender_acc_arr = []

            writer_val.add_scalar('loss', np.mean(loss_arr), epoch)
            save(ckpt_dir=f"{ckpt_dir}/{today_str}/gpu_{gpu_number}",
                 net=net, optim=optim, epoch=epoch, flag=0)

    writer_train.close()
    wrtier_val.close()
    cleanup()


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_number

    ngpus_per_node = torch.cuda.device_count()
    world_size = ngpus_per_node
    mp.spawn(main_worker,
             args=(world_size,),
             nprocs=world_size,
             join=True)

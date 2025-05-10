import sys
sys.path.append("../../../0_CNN_total_Pytorch_new/")
import os
from glob import glob
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from glob import glob
import ast
import cv2
from matplotlib import pyplot as plt
from src.data_set.segmentation import SegDataset
from src.data_set.utils import read_json_as_dict
import random
import pandas as pd
import SimpleITK as sitk
from torch import nn
from natsort import natsorted
import segmentation_models_pytorch as smp
import torch.optim as optim
from src.model.train_util.scheduler import OneCycleLR

import math
from copy import deepcopy
from src.loss.seg_loss import get_dice_score, get_loss_fn, get_bce_loss_class, accuracy_metric
from itertools import chain
from utils import get_config
from src.util.deepspeed import average_across_gpus, toggle_grad, get_deepspeed_config_dict
from src.model.train_util.logger import CSVLogger
import torch.distributed as dist
import argparse
parser = argparse.ArgumentParser(description="Accept loss name")
parser.add_argument("--loss_str", type=str, default="none", required=True, help="train loss str")  # 필수 위치 인자
args, unknown = parser.parse_known_args()
loss_select = args.loss_str
task_name_list = ["drive", "lits_liver", "lits_tumor"]
test_model = "unet"
task_name = "lits_tumor"
# loss_list = ["propotional_bce"]
# "dice", "tversky", "",dice_bce, "propotional",

total_epoch = 100
stage_coef_list = [5, 65]
decay_epoch = total_epoch - sum(stage_coef_list) 
decay_dropout_ratio = 0.25 ** (1  / (total_epoch - sum(stage_coef_list)))

size_list = [512]
batch_size_list = [16, 32]

include_clahe = False
in_channels = 1
num_classes = 2
batch_size = 32
size = 512
get_class = False
get_recon = False
use_seg_in_recon = False
num_gpu = torch.cuda.device_count()
data_num_put_in_one_gpu = 8
loader_batch_size = data_num_put_in_one_gpu * num_gpu
num_workers = 8
# min(loader_batch_size * 2, 16)
on_memory = False

if task_name == "drive":
    batch_size = 4
    loader_batch_size = 2 * num_gpu
    split_json_path =  "../data/0_drive/0_source_data/split.json"
    split_info_dict = read_json_as_dict(split_json_path)
    train_image_path_list, train_mask_path_list = zip(*split_info_dict['train'])
    valid_image_path_list, valid_mask_path_list = zip(*split_info_dict['valid'])

elif task_name in ["lits_liver","lits_tumor"]:
    split_json_path = "../data/1_lits/split.json"
    split_info_dict = read_json_as_dict(split_json_path)
    train_image_path_list, train_mask_path_list = zip(*split_info_dict['train'])
    valid_image_path_list, valid_mask_path_list = zip(*split_info_dict['valid'])
        

train_image_path_list, train_mask_path_list = natsorted(train_image_path_list), natsorted(train_mask_path_list)
valid_image_path_list, valid_mask_path_list = natsorted(valid_image_path_list), natsorted(valid_mask_path_list)

target_size = (size, size)
imread_dict, image_channel_dict, preprocess_dict = get_config(task_name)
augmentation_proba = 0.75
augmentation_policy_dict = {
    "positional": True,
    "noise": True,
    "elastic": True,
    "randomcrop": False,
    "brightness_contrast": True,
    "hist": False,
    "color": False,
    "to_jpeg": False
}
common_arg_dict = {"augmentation_policy_dict": augmentation_policy_dict,
                "target_size": target_size,
                "imread_policy": imread_dict,
                "image_channel_dict": image_channel_dict,
                "preprocess_dict": preprocess_dict,
                "interpolation": "bilinear",
                "dtype": torch.float32,
                }

train_dataset = SegDataset(image_path_list=train_image_path_list,
                            mask_path_list=train_mask_path_list,
                            on_memory=on_memory,
                            augmentation_proba=augmentation_proba,
                        **common_arg_dict
                        )

print("Train dataset initialized.")
print(f"Number of images: {len(train_dataset.image_path_list)}")
print(f"Number of masks: {len(train_dataset.mask_path_list)}")
image_array_0, mask_array_0 = train_dataset[0]
print("Image array shape:", image_array_0.shape)
print("Mask array shape:", mask_array_0.shape)

val_dataset = SegDataset(image_path_list=valid_image_path_list,
                            mask_path_list=valid_mask_path_list,
                            on_memory=on_memory,
                            augmentation_proba=0,
                        **common_arg_dict
                        )

print(f"task_name: {task_name}")
print(f"loss: {loss_select}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

base_model = smp.Unet(
encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
classes=2,                      # model output channels (number of classes in your dataset)
activation="softmax"
)

base_model.encoder.load_state_dict(torch.load("resnet50.pth"))

base_model.encoder.conv1 = nn.Conv2d(1, 64,
                                      kernel_size=7, stride=2, padding=3, bias=False)

if num_gpu > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(base_model)
else: 
    model = base_model

from torch.utils.checkpoint import checkpoint
from torch import nn
from torch.nn.modules.conv import _ConvNd

class CheckpointWrapperLayer(nn.Module):
    def __init__(self, module, use_checkpoint):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint
    def forward(self, *args, **kwargs):
        if self.use_checkpoint:
            output = checkpoint(self.module, *args, **kwargs,
                                use_reentrant=False)
        else:
            output = self.module(*args, **kwargs)

        return output

def add_checkpoint_to_module(model, use_checkpoint=True):
    modules = list(model.named_modules())
    while modules:
        name, module = modules.pop()
        name_list = name.split(".")
        if isinstance(module, _ConvNd) or isinstance(module, nn.Linear):
            target_module = model
            for name_each in name_list[:-1]:
                target_module = getattr(target_module, name_each)
            setattr(target_module, name_list[-1], CheckpointWrapperLayer(module, use_checkpoint))
if num_gpu == 2:
    add_checkpoint_to_module(model, True)
else:
    add_checkpoint_to_module(model, False)

device = torch.device("cuda")
model = model.to(device)
dtype = next(model.parameters()).dtype
print(dtype)
model_param_num = sum(p.numel() for p in base_model.parameters())
print(f"model_param_num = {model_param_num}")

get_l1_loss = nn.L1Loss()
get_l2_loss = nn.MSELoss()
get_class_loss = get_bce_loss_class
get_loss = get_loss_fn(loss_select)

def get_recon_loss_follow_seg(y_recon_pred, y_recon_gt, y_seg_pred):
    recon_image_channel = y_recon_pred.size(1)
    y_seg_pred_weight = 2 * torch.sigmoid(25 * y_seg_pred[:, 1]) - 1
    y_seg_pred_weight = y_seg_pred_weight.unsqueeze(1).repeat(1, recon_image_channel, 1, 1, 1)
    recon_loss = torch.abs(y_recon_pred - y_recon_gt) * y_seg_pred_weight
    return torch.mean(recon_loss)
if use_seg_in_recon:
    get_recon_loss = get_recon_loss_follow_seg
else:
    get_recon_loss = lambda y_recon_pred, y_recon_gt, _: get_l1_loss(y_recon_pred, y_recon_gt)

log_path = f"../result/{task_name}_{test_model}_{loss_select}_{batch_size}"
if get_class:
    log_path = f"{log_path}_class"
if get_recon:
    if use_seg_in_recon:
        log_path = f"{log_path}_recon_with_seg"
    else:
        log_path = f"{log_path}_recon"
os.makedirs(f"{log_path}/weights", exist_ok=True)

epoch_col = ["epoch"]
train_col = ["loss", "dice_score"]
val_col = ["val_loss", "val_dice_score"]
if get_class:
    train_col.append("accuracy")
    val_col.append("val_accuracy")
if get_recon:
    train_col.append("max_recon_diff")
    val_col.append("val_max_recon_diff")

csv_logger = CSVLogger(f"{log_path}/log.csv", epoch_col + train_col + val_col)

import torch.nn.functional as F

def set_dropout_probability(model, decay_dropout_ratio=0.95):
    for _, module in model.named_modules():
        # 모듈이 Dropout 또는 Dropout2d, Dropout3d라면
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            previous_p = module.p
            new_p = previous_p * decay_dropout_ratio
            module.p = new_p
        
def compute_loss_metric(model, x, y, y_label, get_class, get_recon):
    ######## Compute Loss ########
    if get_class and get_recon:
        y_pred, y_label_pred, y_recon_pred = model(x)
        seg_loss = get_loss(y_pred, y)
        class_loss = get_class_loss(y_label_pred, y_label)
        recon_loss = get_recon_loss(y_recon_pred, x, y_pred)
        loss = seg_loss + class_loss + recon_loss
    elif get_class:
        y_pred, y_label_pred = model(x)
        class_loss = get_class_loss(y_label_pred, y_label)
        seg_loss = get_loss(y_pred, y)
        loss = seg_loss + class_loss
    elif get_recon:
        y_pred, y_recon_pred = model(x)
        seg_loss = get_loss(y_pred, y)
        recon_loss = get_recon_loss(y_recon_pred, x, y_pred)
        loss = seg_loss + recon_loss
    else:
        y_pred = model(x)
        seg_loss = get_loss(y_pred, y)
        loss = seg_loss
    ######## Compute Metric #######
    with torch.no_grad():
        _, y_pred = torch.max(y_pred, dim=1)
        y_pred = F.one_hot(y_pred, num_classes=num_classes)
        y_pred = y_pred.permute(0, 3, 1, 2).contiguous()
        metric_dict = {
            "loss": loss.item(),
            "dice_score": get_dice_score(y_pred, y).item()
        }
        if get_class:
            metric_dict["accuracy"] = accuracy_metric(y_label_pred, y_label).item()
        if get_recon:
            metric_dict["recon_diff"] = recon_loss.item()
    return loss, metric_dict

def update_loss_score_dict(loss_score_dict, get_class, get_recon, metric_dict):
    if metric_dict is not None:
        loss_score_dict["loss_list"].append(metric_dict["loss"])
        loss_score_dict["dice_score_list"].append(metric_dict["dice_score"])
        if get_class:
            loss_score_dict["accuracy_list"].append(metric_dict["accuracy"])
        if get_recon:
            loss_score_dict["recon_diff_list"].append(metric_dict["recon_diff"])

def print_and_save_log(train_loss_score_dict, val_loss_score_dict, csv_logger, epoch, total_epoch, get_class, get_recon):
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

def get_loss_score_dict():
    return {
        "loss_list": [],
        "dice_score_list": [],
        "accuracy_list": [],
        "recon_diff_list": []
    }

def get_processed_data(x, y, device):
    x = x.to(device=device, dtype=dtype) 
    y = y.to(device=device, dtype=torch.long)
    y_label = F.one_hot((y.sum(dim=[1, 2]) > 0).long(), num_classes)
    y = F.one_hot(y, num_classes).permute(0, 3, 1, 2).to(dtype=dtype)
    return x, y, y_label


# Define optimizers for both the generator and discriminator
optimizer = optim.Adam(model.parameters(), lr=2e-5, betas=(0.5, 0.999))
step_size = len(train_loader)
# StepLR 스케줄러 정의
scheduler_params = {
"step_size": step_size,
"first_epoch": stage_coef_list[0],
"second_epoch": stage_coef_list[1],
"total_epoch": total_epoch
}
lr_scheduler = OneCycleLR(optimizer, **scheduler_params)

pbar_fn = tqdm
for epoch in range(1, total_epoch + 1):
    train_pbar = pbar_fn(train_loader)
    toggle_grad(model, require_grads=True)
    model.train()
    train_loss_score_dict = get_loss_score_dict()
    val_loss_score_dict = get_loss_score_dict()
    for batch_idx, (x, y) in enumerate(train_pbar, start=1):
        optimizer.zero_grad()
        x, y, y_label = get_processed_data(x, y, device)
        loss, metric_dict = compute_loss_metric(model, x, y, y_label, get_class, get_recon)
        loss.backward()
        optimizer.step()
        update_loss_score_dict(train_loss_score_dict, get_class, get_recon, metric_dict)
        train_pbar.set_postfix({'Epoch': f'{epoch}/{total_epoch}',
                                'loss': f'{np.mean(train_loss_score_dict["loss_list"]):.4f}',
                                'dice_score': f'{np.mean(train_loss_score_dict["dice_score_list"]):.4f}',
                                'current_loss':f'{train_loss_score_dict["loss_list"][-1]:.4f}'})
        lr_scheduler.step()
    toggle_grad(model, require_grads=False)
    model.eval()
    with torch.no_grad():
        valid_pbar = pbar_fn(val_loader)
        for batch_idx, (x, y) in enumerate(valid_pbar):
            x, y, y_label = get_processed_data(x, y, device)
            loss, metric_dict = compute_loss_metric(model, x, y, y_label, get_class, get_recon)
            update_loss_score_dict(val_loss_score_dict, get_class, get_recon, metric_dict)

    if epoch >= sum(stage_coef_list):
        set_dropout_probability(model, decay_dropout_ratio=decay_dropout_ratio)


    print_and_save_log(train_loss_score_dict, val_loss_score_dict, csv_logger, epoch, total_epoch, get_class, get_recon)
    torch.save(model.state_dict(), f"{log_path}/weights/{epoch:03}.ckpt")
    
torch.cuda.empty_cache()
del model
del base_model
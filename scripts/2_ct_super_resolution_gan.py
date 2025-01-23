#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pydicom
from glob import glob
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import os
import shutil
import SimpleITK as sitk
import time
from natsort import natsorted
from matplotlib import pyplot as plt
from collections import Counter
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
import SimpleITK as sitk


# In[2]:


import os
# gpu_on = True
# gpu_number = "0,1,2"
# os.environ['CUDA_VISIBLE_DEVICES'] = gpu_number
import sys
sys.path.append("../../../CNN_total_Pytorch")
from src.data_set.utils import get_parent_dir_name

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial
import random
import pydicom
from albumentations import Compose
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import SimpleITK as sitk
from glob import glob
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from natsort import natsorted
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
import nibabel as nib
import json
from time import sleep
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from src.model.train_util.common import mask_gradient
from src.model.train_util.wgan_blend import compute_gradient_penalty_blend, get_blend_images_2d
from src.model.train_util.wgan import compute_gradient_penalty
from src.util.fold_unfold import extract_patch_tensor, combine_region_voting_patches, combine_region_voting_patches_with_patch_weights, PatchSplitModel
from src.util.crop_batch import rotate_and_crop_tensor
from src.model.train_util.logger import CSVLogger
from src.model.swin_transformer.model_2d.multi_task_v5 import SwinMultitask
from src.model.inception_resnet_v2.multi_task.multi_task_2d_v2 import InceptionResNetV2MultiTask2D
import torch

from dataset import CustomImageDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def check_is_uid_same(sparse_folder, dense_folder):
    sparse_uid = get_parent_dir_name(sparse_folder, 1)
    dense_uid = get_parent_dir_name(dense_folder, 1)
    return sparse_uid == dense_uid

# Function to extract StudyDate from a DICOM file
def get_study_date(dicom_folder):
    dicom_files = glob(os.path.join(dicom_folder, "*.dcm"))
    if not dicom_files:
        return None
    # Load the first DICOM file in the folder
    dcm = pydicom.dcmread(dicom_files[0])
    if "StudyDate" in dcm:
        return dcm.StudyDate
    else:
        return None
    
# DICOM key dictionary
dicom_keys = {
    "InstitutionName": ("0008", "0080"),
    "PatientAge": ("0010", "1010"),
    "PatientSex": ("0010", "0040"),
    "StudyDate": ("0008", "0020"),  # StudyDate is used to split dataset
    "Manufacturer": ("0008", "1090"),
    "ConvolutionKernel": ("0018", "1210"),
    "SliceThickness": ("0018", "0050"),
    "SpacingBetweenSlices": ("0018", "0088")
}


sparse_folder_list = natsorted(glob("../data/2. resampled_data/*/*"))
dense_folder_list = natsorted(glob("../data/1. selected_data/*/*"))

sparse_folder_list = [folder for folder in sparse_folder_list if os.path.isdir(folder)]
dense_folder_list = [folder for folder in dense_folder_list if os.path.isdir(folder)]

series_uid_is_same_list = [check_is_uid_same(sparse_folder, dense_folder)
                          for sparse_folder, dense_folder in zip(sparse_folder_list, dense_folder_list)]

for idx, value in enumerate(series_uid_is_same_list):
    if not value:
        print(f"Inconsistency at index {idx}")

assert all(series_uid_is_same_list), "check series uid consistency"

# Extract StudyDate for each folder
study_dates = [get_study_date(folder) for folder in sparse_folder_list]

# Ensure non-null values and combine with the folder paths
total_data = [(date, sparse, dense) for date, sparse, dense in zip(study_dates, sparse_folder_list, dense_folder_list) if date]

# Sort by StudyDate
total_data = sorted(total_data, key=lambda x: x[0], reverse=True)  # Sort in descending order (recent first)


data_num = len(total_data)
test_num = round(data_num * 0.10)
test_data = total_data[:test_num]
test_sparse_folder_list = [item[1] for item in test_data]
test_dense_folder_list = [item[2] for item in test_data]

remaining_data = total_data[test_num:]

image_size = 512



train_ratio = 0.89
valid_ratio = 0.11
test_ratio = 0.0
get_blend = True
get_real_fake = True
use_wgan = True
use_mask_gradient = False
use_patch = False
half_patch_size = False
use_cnn = True
exclude_edge_slice = False
if exclude_edge_slice:
    output_slice_num = 2
else:
    output_slice_num = 4
# Number of epochs
load_epoch = 0
num_epochs = 100
min_ignore_prob = 0.5
max_ignore_prob = 0.95
min_coef = 100
max_coef = 1000
num_gpu = torch.cuda.device_count()

if half_patch_size:
    patch_size = image_size // 8
else:
    patch_size = image_size // 4
stride = patch_size // 2
pad_size = stride

if use_patch:
    image_size_tuple = (patch_size, patch_size)
    embed_dim = 32
    num_heads = [1, 2, 4, 8]
    block_size = 8
    blend_patch_size = 16
    if half_patch_size:
        validity_shape = (4, 4)
    else:
        validity_shape = (8, 8)
else:
    image_size_tuple = (image_size, image_size)
    embed_dim_gen = 96  # generator 모델의 embed_dim 값
    embed_dim_disc = 96  # discriminator 모델의 embed_dim 값
    num_heads = [2, 4, 8, 16]
    num_heads_disc = [2, 4, 8]
    block_size = 16
    blend_patch_size = 64
    validity_shape = (8, 8)

train_info_dict_path = "./train_info.json"

if os.path.exists(train_info_dict_path):
    with open(train_info_dict_path, 'r') as json_file:
        train_info_dict = json.load(json_file)
    train_idx_list = train_info_dict["train"]
    valid_idx_list = train_info_dict["valid"]    
else:
    remaining_data_num = len(remaining_data)
    train_num = round(remaining_data_num * train_ratio)

    random_idx_list = list(range(remaining_data_num))
    random.shuffle(random_idx_list)

    train_idx_list = sorted(random_idx_list[:train_num])
    valid_idx_list = sorted(random_idx_list[train_num:])
    
    train_info_dict = {
        "train": train_idx_list,
        "valid": valid_idx_list
    }
    with open(train_info_dict_path, 'w') as json_file:
        json.dump(train_info_dict, json_file, indent=4)    

print(train_idx_list)
train_data = [remaining_data[idx] for idx in train_idx_list]
valid_data = [remaining_data[idx] for idx in valid_idx_list]
        
train_sparse_folder_list = [item[1] for item in train_data]
train_dense_folder_list = [item[2] for item in train_data]
valid_sparse_folder_list = [item[1] for item in valid_data]
valid_dense_folder_list = [item[2] for item in valid_data]


print(f"Train: {len(train_sparse_folder_list)}")
print(f"Valid: {len(valid_sparse_folder_list)}")
print(f"Test: {len(test_sparse_folder_list)}")


# In[6]:


train_dataset = CustomImageDataset(train_sparse_folder_list, train_dense_folder_list, exclude_edge_slice=exclude_edge_slice)
val_dataset = CustomImageDataset(valid_sparse_folder_list, valid_dense_folder_list, exclude_edge_slice=exclude_edge_slice)

train_dataloader = DataLoader(train_dataset, batch_size=num_gpu, pin_memory=True,
                             num_workers=12, shuffle=True)

val_dataloader = DataLoader(val_dataset, batch_size=num_gpu, pin_memory=True,
                             num_workers=12, shuffle=False)


# In[7]:


# In[8]:


with torch.no_grad():
    x, y = train_dataset[0]
    print(x.shape, y.shape)
    for x, y in train_dataloader:
        print(x.shape, y.shape)
        break


# In[9]:

class_channel = 1


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
use_checkpoint = [True, True, True, True]
if use_cnn:
    gen_model = InceptionResNetV2MultiTask2D(input_shape=(2, *image_size_tuple),
                                            class_channel=None, seg_channels=output_slice_num,
                                            validity_shape=(1, 8, 8), inject_class_channel=None,
                                            block_size=16, include_cbam=False, decode_init_channel=None,
                                            norm="instance", act="leakyrelu", dropout_proba=0.0,
                                            seg_act="sigmoid", class_act="softmax", 
                                            recon_act="sigmoid", validity_act="sigmoid",
                                            get_seg=True, get_class=False, get_recon=False, get_validity=False,
                                            use_class_head_simple=False, include_upsample=False,
                                            use_decode_simpleoutput=True, use_seg_conv_transpose=False,
                                            use_checkpoint=False).to(device)
else:
    gen_model = SwinMultitask(img_size=image_size_tuple, patch_size=1,
                                    in_chans=2, num_classes=None, seg_out_chans=output_slice_num,
                                    norm_layer="instance", act_layer="silu",
                                    embed_dim=embed_dim_gen, depths=[2, 2, 2, 1], num_heads=num_heads,
                                    window_sizes=[8, 4, 2, 2], mlp_ratio=4., qkv_bias=True, ape=True,
                                    drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                                    class_act="softmax", seg_out_act="sigmoid", validity_act="sigmoid",
                                        get_class=False, get_seg=True, get_validity=False,
                                    use_checkpoint=use_checkpoint
                                        ).to(device)


disc_model = InceptionResNetV2MultiTask2D(input_shape=(output_slice_num, *image_size_tuple), 
                                          class_channel=None, seg_channels=None, validity_shape=(1, *validity_shape),
                                          inject_class_channel=None,
                                         block_size=block_size, include_cbam=False, decode_init_channel=None,
                                          norm="instance", act="relu", dropout_proba=0.05,
                                         class_act="softmax", seg_act="sigmoid", 
                                          validity_act=None if use_wgan else "sigmoid",
                                         get_seg=False, get_class=False, get_validity=True,
                                         use_class_head_simple=False, include_upsample=False,
                                         use_decode_simpleoutput=True, use_seg_conv_transpose=False,
                                         use_checkpoint=False).to(device)
print(count_parameters(gen_model))
print(count_parameters(disc_model))


# In[10]:


import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from src.loss.ssim_psnr import get_ssim, get_psnr, get_max_ssim_loss_fn

if num_gpu > 1:
    print(f"using {torch.cuda.device_count()} GPUs!")
    gen_model = nn.DataParallel(gen_model)
    disc_model = nn.DataParallel(disc_model)
    
gen_criterion = torch.nn.L1Loss()  #중복 정의!
maxpool = torch.nn.MaxPool2d(64, stride=64)

if use_mask_gradient:
    def get_max_loss_fn(y_pred, y_true, data_range=1.0, filter_fn="avg", version="skimages", image_to_patch_ratio=4):
        batch_size, _, *image_size_list = y_pred.shape
        target_mean_dim_tuple = tuple(range(1, 2 + len(image_size_list)))
        image_size = int(image_size_list[-1])
        patch_size = image_size // image_to_patch_ratio
        stride = patch_size // 4
        num_patch = int((image_size - patch_size) / stride) + 3
        
        # y_pred_patch.shape = [B * (num_patch ** 2), C, H, W]
        y_pred_patch = extract_patch_tensor(y_pred, patch_size, stride, pad_size=stride)
        y_true_patch = extract_patch_tensor(y_true, patch_size, stride, pad_size=stride)

        ssim_loss_patch = -get_ssim(y_pred_patch, y_true_patch, data_range=data_range, reduction="none",
                                    filter_fn=filter_fn, version=version)
        # ssim_min_k_score_patch.shape = [B, num_patch]
        ssim_loss_patch = ssim_loss_patch.view(batch_size, num_patch ** 2)
        max_ssim_k_loss_patch  = torch.topk(ssim_loss_patch, num_patch, dim=1, largest=True).values
        max_ssim_loss = max_ssim_k_loss_patch.mean()
        
        l1_loss_patch = F.l1_loss(y_pred_patch, y_true_patch, reduction="none").mean(target_mean_dim_tuple)
        l1_loss_patch = l1_loss_patch.view(batch_size, num_patch ** 2)
        max_l1_loss_patch = torch.topk(l1_loss_patch, num_patch, dim=1, largest=True).values
        # ssim_min_k_score_patch.shape = [B, num_patch, C, H, W]
        max_l1_loss = max_l1_loss_patch.mean()

        psnr_loss_patch = -get_psnr(y_pred_patch, y_true_patch, data_range=data_range, reduction="none") / 50
        psnr_loss_patch = psnr_loss_patch.view(batch_size, num_patch ** 2)
        psnr_max_k_loss_patch = torch.topk(psnr_loss_patch, num_patch, dim=1, largest=True).values
        # ssim_min_k_score_patch.shape = [B, num_patch]
        max_pnsr_loss = psnr_max_k_loss_patch.mean()
        return max_ssim_loss, max_l1_loss, max_pnsr_loss

    def gen_criterion(y_pred, y_true):
        l1_loss = F.l1_loss(y_pred, y_true)
        y_pred_maxpool = maxpool(y_pred)
        y_true_maxpool = maxpool(y_true)
        
        max_loss = torch.abs(y_pred_maxpool - y_true_maxpool).mean()
        ssim_loss = -get_ssim(y_pred, y_true, filter_fn="avg", version="skimages")
        max_ssim_loss, max_l1_loss, max_pnsr_loss  = get_max_loss_fn(y_pred, y_true)
        total_loss = (l1_loss * 5 + ssim_loss * 1 + max_ssim_loss * 1 + max_l1_loss * 1 + max_pnsr_loss * 1 + max_loss * 1) / 10
    #     total_loss = (l1_loss * 7 + ssim_loss * 2 + max_loss * 1) / 10
        return total_loss, l1_loss, ssim_loss, max_loss
else:
    def gen_criterion(y_pred, y_true):
        l1_loss = F.l1_loss(y_pred, y_true)
        y_pred_maxpool = maxpool(y_pred)
        y_true_maxpool = maxpool(y_true)
        
        max_loss = torch.abs(y_pred_maxpool - y_true_maxpool).mean()
        ssim_loss = -get_ssim(y_pred, y_true, filter_fn="avg", version="skimages")
        total_loss = (l1_loss * 7 + ssim_loss * 2 + max_loss * 1) / 10
        return total_loss, l1_loss, ssim_loss, max_loss

disc_criterion = torch.nn.MSELoss()

# In[11]:


from src.model.train_util.common import clip_gradients
from src.model.train_util.scheduler import OneCycleLR
import torch.optim as optim

def get_is_decay_lr(current_learning_rate):
    if current_learning_rate < 1e-5:
        return False
    else:
        return True
def get_current_lr(optimizer):
    lr = optimizer.param_groups[0]['lr']
    return lr

def plot_result(real_images, target_images, fake_images, exclude_edge_slice, use_patch):
    batch_size = real_images.shape[0]
    if use_patch:
        target_images = combine_region_voting_patches_with_patch_weights(target_images, batch_size=num_gpu,
                                                                        image_size=image_size, patch_size=patch_size, stride=stride,
                                                                        pad_size=stride, img_dim=2)
        fake_images = combine_region_voting_patches_with_patch_weights(fake_images, batch_size=num_gpu,
                                                                        image_size=image_size, patch_size=patch_size, stride=stride,
                                                                        pad_size=stride, img_dim=2)

    real_image = real_images[0].detach().cpu().numpy().transpose(1, 2, 0)
    target_image = target_images[0].detach().cpu().numpy().transpose(1, 2, 0)
    fake_image = fake_images[0].detach().cpu().numpy().transpose(1, 2, 0)
    
    if exclude_edge_slice:
        plot_slice = 2
    else:
        plot_slice = 4

    fig, ax = plt.subplots(2, plot_slice, figsize=(8 * plot_slice, 16))
    for idx in range(plot_slice):
        ax[0, idx].imshow(target_image[..., idx], 
                        cmap="gray", vmin=0, vmax=1)
        ax[1, idx].imshow(fake_image[..., idx], 
                        cmap="gray", vmin=0, vmax=1)
    plt.savefig(f"./{plots_folder}/polt_{epoch:03d}.png")
    plt.tight_layout()
    plt.show()

    # blend_image = blend_images[0].detach().cpu().numpy().transpose(1, 2, 0)
    # blend_label = blend_label[0, 0].detach().cpu().numpy()
    # blend_disc_pred = blend_outputs[0, 0].detach().cpu().numpy()
        
    # fig, ax = plt.subplots(1, 4, figsize=(32, 8))
    # ax[0].imshow(blend_image[..., 0], 
    #              cmap="gray", vmin=0, vmax=1)
    # ax[1].imshow(blend_image[..., -1], 
    #              cmap="gray", vmin=0, vmax=1)
    # ax[2].imshow(blend_label)
    # ax[3].imshow(blend_disc_pred)
    # for i in range(blend_disc_pred.shape[0]):
    #     for j in range(blend_disc_pred.shape[1]):
    #         ax[3].text(j, i, format(blend_disc_pred[i, j], '.2f'), 
    #                       ha="center", va="center", color="red", fontsize=16)
    # plt.tight_layout()
    # plt.show()
    
def change_model_grad_parameters(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad

def get_folder_names(get_real_fake, get_blend, use_wgan, use_mask_gradient, use_patch, half_patch_size, use_cnn, exclude_edge_slice):
    if get_real_fake and get_blend:
        folder_name = "real_fake_blend"
    elif get_real_fake and not get_blend:
        folder_name = "real_fake"
    elif not get_real_fake and get_blend:
        folder_name = "blend"
    else:
        folder_name = "neither_real_fake_nor_blend"
    
    if use_wgan:
        folder_name = f"{folder_name}_wgan"
    else:
        folder_name = f"{folder_name}_lsgan"
        
    if use_mask_gradient:
        folder_name = f"{folder_name}_mask_gradient"
        
    if use_patch:
        if half_patch_size:
            folder_name = f"{folder_name}_use_half_patch"
        else:
            folder_name = f"{folder_name}_use_patch"
    if use_cnn:
        folder_name = f"{folder_name}_cnn"

    if exclude_edge_slice:
        folder_name = f"{folder_name}_exclude_edge_slice"
    weights_folder = f"./weights/{folder_name}"
    plots_folder = f"./plots/{folder_name}"
    log_csv_folder = "./logs"
    log_csv_path = f"{log_csv_folder}/{folder_name}.csv"
    # Ensure the directories exist
    os.makedirs(weights_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)
    os.makedirs(log_csv_folder, exist_ok=True)
    
    return weights_folder, plots_folder, log_csv_path


# In[14]:


weights_folder, plots_folder, log_csv_path = get_folder_names(get_real_fake, get_blend, use_wgan, use_mask_gradient, 
                                                              use_patch, half_patch_size, use_cnn, exclude_edge_slice)

epoch_col = ["epoch"]
train_col = ["disc_blend", "disc_real_fake", "disc_loss", "disc_gp",
             "gen_image", "gen_ssim_mean", "gen_ssim_std", "gen_disc"]
val_col = ["val_avg_ssim", "val_std_ssim", "val_avg_psnr", "val_std_psnr"]

csv_logger = CSVLogger(log_csv_path, epoch_col + train_col + val_col)

# Define optimizers for both the generator and discriminator
optimizer_gen = optim.Adam(gen_model.parameters(), lr=4e-5, betas=(0.5, 0.999))
optimizer_disc = optim.Adam(disc_model.parameters(), lr=4e-5, betas=(0.5, 0.999))

step_size = len(train_dataloader)  # 매 30 step마다

# StepLR 스케줄러 정의
scheduler_params = {
"step_size": step_size,
"first_epoch": 2,
"second_epoch": 68,
"total_epoch": num_epochs
}
gen_scheduler = OneCycleLR(optimizer_gen, **scheduler_params)
disc_scheduler = OneCycleLR(optimizer_disc, **scheduler_params)
if load_epoch > 0:
    gen_state_dict = torch.load(f'./{weights_folder}/generator_epoch_{load_epoch:03d}.pth')
    disc_state_dict = torch.load(f'./{weights_folder}/discriminator_epoch_{load_epoch:03d}.pth')
    gen_model.load_state_dict(gen_state_dict["model"])
    disc_model.load_state_dict(disc_state_dict["model"])
    optimizer_gen.load_state_dict(gen_state_dict["optimizer"])
    optimizer_disc.load_state_dict(disc_state_dict["optimizer"])
    gen_scheduler.load_state_dict(gen_state_dict["scheduler"])
    disc_scheduler.load_state_dict(disc_state_dict["scheduler"])

coef_step_size = num_epochs * step_size
is_decay_lr = get_is_decay_lr(get_current_lr(optimizer_gen))

for epoch in range(load_epoch + 1, num_epochs + 1):
    gen_model.train()
    disc_model.train()
    loop = tqdm(train_dataloader, total=len(train_dataloader), leave=True)
    loss_disc_list = []
    loss_gen_list = []
    loss_gen_image_list = []
    loss_gen_disc_list = []
    if epoch <= 1000:
        clip_outlier = False
    else:
        clip_outlier = True
    
    disc_blend_list = []
    disc_real_fake_list = []
    disc_loss_list = []
    disc_gp_list = []
    gen_image_list = []
    gen_ssim_list = []
    gen_disc_list = []

    for idx, (real_images, target_images) in enumerate(loop):
        current_step_size = (epoch - 1) * step_size + idx
        current_coef = max_coef + (min_coef - max_coef) * (current_step_size / coef_step_size)
        # Moving tensors to the configured device
        # real_images = [B, 2, 512, 512]
        # fake_images = [B, 4, 512, 512]
        ignore_prob = max_ignore_prob + (min_ignore_prob - max_ignore_prob) * (current_step_size / coef_step_size)
        
        change_model_grad_parameters(gen_model, False)
        change_model_grad_parameters(disc_model, True)
        optimizer_disc.zero_grad()
        real_images = real_images.to(device)
        target_images = target_images.to(device)

        if use_patch:
            real_images = extract_patch_tensor(real_images, patch_size, stride, pad_size=stride)
            target_images = extract_patch_tensor(target_images, patch_size, stride, pad_size=stride)

        real_disc_input = target_images
        # -----------------------
        # 1. Update Discriminator
        # -----------------------
        fake_images = gen_model(real_images)
        fake_disc_input = fake_images
        loss_disc = 0
        divisor = 0
        lambda_gp = 10.0
        gradient_penalty_sum = torch.tensor(0)
        # Real images
        
        blend_loss = torch.tensor(0)
        real_fake_loss = torch.tensor(0)
        if get_blend:
            blend_images, blend_label = get_blend_images_2d(target_images, fake_images, patch_size=blend_patch_size)
            #if use_mask_gradient:
            #    mask_gradient(target_images, blend_images, ignore_prob=ignore_prob)
            blend_outputs = disc_model(blend_images)
               
            # Combine losses
            real_parts = blend_outputs[blend_label]
            fake_parts = blend_outputs[~blend_label]
            if use_wgan:
                gradient_penalty_blend = compute_gradient_penalty_blend(disc_model, target_images, fake_images, blend_label)
                gradient_penalty_sum = gradient_penalty_sum + gradient_penalty_blend
                mean_real = torch.mean(real_parts)
                mean_fake = torch.mean(fake_parts)
                blend_loss = -mean_real + mean_fake
            else:
                loss_disc_real = disc_criterion(real_parts, torch.ones_like(real_parts))
                loss_disc_fake = disc_criterion(fake_parts, torch.zeros_like(fake_parts))
                blend_loss = (loss_disc_real + loss_disc_fake) / 2
                divisor += 1
            loss_disc = loss_disc + blend_loss
        if get_real_fake:
            outputs_real = disc_model(real_disc_input)
            outputs_fake = disc_model(fake_images.detach())
            
            if use_wgan:
                gradient_penalty = compute_gradient_penalty(disc_model, target_images, fake_images)
                gradient_penalty_sum = gradient_penalty_sum + gradient_penalty
                mean_real = torch.mean(outputs_real)
                mean_fake = torch.mean(outputs_fake)
                real_fake_loss = -mean_real + mean_fake
            else:
                real_label = torch.ones(outputs_real.size()).to(device)
                fake_label = torch.zeros(outputs_fake.size()).to(device)
                loss_real = disc_criterion(outputs_real, real_label)
                loss_fake = disc_criterion(outputs_fake, fake_label)
                real_fake_loss = loss_real + loss_fake
                divisor += 2
            loss_disc = loss_disc + real_fake_loss + gradient_penalty_sum * lambda_gp
            
        if not use_wgan:
            loss_disc = loss_disc / divisor
        
        loss_disc.backward()
#         clip_gradients(disc_model)
        optimizer_disc.step()
        # --------------------
        # 2. Update Generator
        # --------------------
        change_model_grad_parameters(gen_model, True)
        change_model_grad_parameters(disc_model, False)
        optimizer_gen.zero_grad()
        fake_images = gen_model(real_images)
        #if use_mask_gradient:
        #    mask_gradient(target_images, fake_images, ignore_prob=ignore_prob)
        outputs = disc_model(fake_images)
        
        loss_gen_image, l1_loss, ssim_loss, max_loss = gen_criterion(fake_images, target_images)
        loss_gen_disc = disc_criterion(outputs, torch.ones_like(outputs))
        loss_gen = (loss_gen_image * current_coef + loss_gen_disc) / (current_coef + 1)
        loss_gen.backward()
#         clip_gradients(gen_model)
        optimizer_gen.step()
        loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
        loop.set_postfix(Disc_Loss=loss_disc.item(), Gen_Image_Loss=l1_loss.item(), Gen_Disc_Loss=loss_gen_disc.item())
        loss_disc_list.append(loss_disc.item())
        loss_gen_list.append(loss_gen.item())
        loss_gen_image_list.append(loss_gen_image.item())
        loss_gen_disc_list.append(loss_gen_disc.item())
        if is_decay_lr:
            gen_scheduler.step()
            disc_scheduler.step()
            current_learning_rate = optimizer_gen.param_groups[0]['lr']
            is_decay_lr = get_is_decay_lr(current_learning_rate)
        
        disc_blend_scalar = blend_loss.item()
        disc_real_fake_scalar = real_fake_loss.item()
        disc_loss_scalar = loss_disc.item()
        disc_gp_scalar = gradient_penalty_sum.item()
        gen_image_scalar = loss_gen_image.item()
        gen_ssim_scalar = -ssim_loss.item()
        gen_disc_scalar = loss_gen_disc.item()
        
        disc_blend_list.append(disc_blend_scalar)
        disc_real_fake_list.append(disc_real_fake_scalar)
        disc_loss_list.append(disc_loss_scalar)
        disc_gp_list.append(disc_gp_scalar)
        gen_image_list.append(gen_image_scalar)
        gen_ssim_list.append(gen_ssim_scalar)
        gen_disc_list.append(gen_disc_scalar)

#         if idx % 100 == 0:
#             plot_result(real_images, target_images, fake_images, blend_images, blend_label, blend_outputs)
    print(f"Epoch [{epoch}/{num_epochs}] Loss D: {np.mean(loss_disc_list):.3f}, Loss G: {np.mean(loss_gen_list):.3f}, LR: {current_learning_rate}")
    print(f"Epoch [{epoch}/{num_epochs}] Loss Gen Image: {np.mean(loss_gen_image_list):.3f}, Loss Gen Disc: {np.mean(loss_gen_disc_list):.3f}")
    
    log_str_list = [f"{epoch}"]
    train_str_list = [f'{np.mean(disc_blend_list):.4f}',
                      f'{np.mean(disc_real_fake_list):.4f}',
                      f'{np.mean(disc_loss_list):.4f}',
                      f'{np.mean(disc_gp_list):.4f}',
                      f'{np.mean(gen_image_list):.4f}',
                      f'{np.mean(gen_ssim_list):.4f}',
                      f'{np.std(gen_ssim_list):.4f}',
                      f'{np.mean(gen_disc_list):.4f}']
    log_str_list = log_str_list + train_str_list
    # -----------------
    # 3. Save Weights
    # -----------------
    gen_model.eval()
    disc_model.eval()
    ssim_scores = []
    psnr_scores = []
    with torch.no_grad():
        for real_images, target_images in val_dataloader:
            real_images = real_images.to(device)
            target_images = target_images.to(device)
            if use_patch:
                real_images = extract_patch_tensor(real_images, patch_size, stride, pad_size=stride)
                target_images = extract_patch_tensor(target_images, patch_size, stride, pad_size=stride)
#             if get_blend:
#                 blend_images, blend_label = get_blend_images_2d(target_images, fake_images)
#                 blend_outputs = disc_model(blend_images)
            fake_images = gen_model(real_images)
            ssim_score = get_ssim(fake_images, target_images, filter_fn="avg", version="skimages")
            psnr_score = get_psnr(fake_images, target_images)

            ssim_scores.append(ssim_score.item())
            psnr_scores.append(psnr_score.item())

    avg_ssim = f"{np.mean(ssim_scores):.3f}"
    std_ssim = f"{np.std(ssim_scores):.3f}"
    avg_psnr = f"{np.mean(psnr_scores):.3f}"
    std_psnr = f"{np.std(psnr_scores):.3f}"
    val_str_list = [avg_ssim, std_ssim, avg_psnr, std_psnr]
    log_str_list = log_str_list + val_str_list
    csv_logger.writerow([*log_str_list])

    print(f"Epoch [{epoch}/{num_epochs}] SSIM: {avg_ssim} ± {std_ssim} , PSNR: {avg_psnr} ± {std_psnr} dB")
    plot_result(real_images, target_images, fake_images, exclude_edge_slice, use_patch)
        
    if epoch % 1 == 0: # Save every 10 epochs (change the number as you like)
        torch.save({"model": gen_model.state_dict(), "optimizer": optimizer_gen.state_dict(),
                     "scheduler": gen_scheduler.state_dict()},
                   f'./{weights_folder}/generator_epoch_{epoch:03d}.pth')
        torch.save({"model": disc_model.state_dict(), "optimizer": optimizer_disc.state_dict(),
                     "scheduler": gen_scheduler.state_dict()},
                   f'./{weights_folder}/discriminator_epoch_{epoch:03d}.pth')


# In[ ]:





# In[ ]:





# In[ ]:





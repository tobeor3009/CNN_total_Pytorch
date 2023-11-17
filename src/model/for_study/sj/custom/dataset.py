import matplotlib.pyplot as plt
import time
import numpy as np
import argparse
import math
import os
from glob import glob
import random
from PIL import Image
from tqdm.auto import tqdm

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import transforms
from PIL import Image

from .utils import *
from torchvision import utils
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pydicom


class RealDataset(torch.utils.data.Dataset):
    def __init__(self,
                 split='train',
                 transform=None,
                 use_slice = True,
                 urgency = ["normal"],
                 dcm_path = "Brain_CT_Data/train/dicom"):
        self.split          = split
        self.transform      = transform
        self.urgency        = urgency
        # self.imgs           = load_obj("normal_lst.pkl")
        # self.patients = glob(os.path.join("/mnt/nas125/SeungjunLee/Shares/brainct_8bit_normal_rgb_png","*"))   
        imgs = {label: glob(os.path.join(dcm_path, label, "enrolled", "*", "*.png"))
                     for label in self.urgency}

        self.imgs = []
        for label in self.urgency:
            self.imgs += imgs[label][:-10000] if self.split == "train" else imgs[label][-10000:]        
        random.shuffle(self.imgs)

        if self.split == "train":
            if use_slice:
                self.imgs = random.choices(self.imgs, k=100000)
            if "normal" in self.urgency:
                self.imgs += glob(os.path.join("Brain_CT_Data", "test", "*","2021031*", "Normal", "*", "png", "*.png"))
            if "benign" in self.urgency:
                self.imgs += glob(os.path.join("Brain_CT_Data", "test", "*","2021031*", "Benign", "*", "png", "*.png"))
            if "emergency" in self.urgency:
                self.imgs += glob(os.path.join("Brain_CT_Data", "test", "*","2021031*", "Emergency", "*", "png", "*.png"))
            if "urgent" in self.urgency:
                self.imgs += glob(os.path.join("Brain_CT_Data", "test", "*","2021031*", "Urgent", "*", "png", "*.png"))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        label = 1 if "Normal" in self.imgs[idx] or "normal" in self.imgs[idx] else -1     
        # img = dcm2np(self.imgs[idx], windowing=True)

        if self.transform:
            img = self.transform(img)

        return img, label

class AxialScan(torch.utils.data.Dataset):
    def __init__(self,
                 split='train',
                 transform=None,
                 dcm_path = "../Dataset/Asan_Brain_CT/train/dicom",
                 n_slices = 32,
                 reverse = True):
        self.split          = split
        self.transform      = transform
        self.n_slices       = n_slices
        self.reverse        = reverse

        self.patients = glob(os.path.join(dcm_path, "*", "enrolled", "*"))
        
        n_train = int(len(self.patients) * 0.8)
        n_valid = int(len(self.patients) * 0.1)
        n_test  = len(self.patients) - (n_train + n_valid)

        if self.split == "train":
            self.patients = self.patients[: n_train]
        elif self.split == "valid":
            self.patients = self.patients[n_train: n_train + n_valid]
        elif self.split == "test":
            self.patients = self.patients[-n_test:]

        print(f"{self.split} dataset (n = {len(self.patients)})")
        
    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        imgs, label = get_scan(glob(os.path.join(self.patients[idx], "*.dcm")), reverse = self.reverse)
        if imgs == [] or label == []:
            return __getitem__(random.randint(0, self.__len__()))
        
        # post-processing: image data
        n_slices = len(imgs)
        if self.n_slices <= n_slices:
            start_idx = random.randint(0, n_slices - self.n_slices)
            end_idx = start_idx + self.n_slices
            imgs = imgs[start_idx: end_idx]
        else:
            imgs = (self.n_slices - n_slices) * [imgs[0]] + imgs

        if self.transform:
            imgs = [self.transform(img) for img in imgs]
        imgs = torch.stack(imgs).float()

        # post-processing: meta data
        label = torch.FloatTensor(label).float().unsqueeze(0).repeat(self.n_slices, 1)

        return imgs, label
    
class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, g_module, batch_size, is_latent=True , w_plus = True, truncation = 1, device = "cuda"):
        self.g_module      = g_module
        self.batch_size    = batch_size
        self.is_latent     = is_latent
        self.device        = device
        self.w_plus        = w_plus
        self.truncation    = truncation
        self.n_latent      = g_module.n_latent

        # w_stat = load_obj('history/w_plus_stat.pkl') if self.w_plus else load_obj('history/w_stat.pkl')
        # self.w_mean = w_stat["mean"]
        # self.w_std = w_stat["std"]
        #
        # self.truncation_latent = torch.stack(
        #     [torch.from_numpy(self.w_mean[i]).to(self.device)
        #      for i in range(32)],
        #     dim=0)

        # if self.w_plus:
        #     self.latent_std        = torch.stack(
        #         [torch.tensor(self.w_std[i], dtype=torch.float).to(self.device)
        #          for i in range(32)],
        #         dim = 0)
        # else:
        #     self.latent_std        = torch.stack(
        #         [torch.from_numpy(self.w_std[i]).to(self.device)
        #          for i in range(32)],
        #         dim = 0)

        with torch.no_grad():
            
            n_mean_latent = 10000
            noise_sample = torch.randn(n_mean_latent, 512, device = device)
            latent_out = g_module.style(noise_sample) 
            latent_mean = latent_out.mean(0,keepdim = True)
            latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

            if self.w_plus:
                self.w_mean = latent_mean.unsqueeze(0).repeat(1,self.n_latent,1) # [1,16,512]
                assert self.w_mean.size() == (1,self.n_latent,512), self.w_mean.size()
            else:
                self.w_mean = latent_mean
                assert self.w_mean.size() == (1,512) , self.w_mean.size()

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        # 1. generate noise
        # 2. map noise to style(or latent)
        # 3. create from noise
        with torch.no_grad():
            if self.w_plus:
                "random initialize latent code"
                z1  = torch.randn([self.batch_size, 512], device=self.device)
                z2  = torch.randn([self.batch_size, 512], device=self.device)

                w1 = self.g_module.style(z1)
                w2 = self.g_module.style(z2)

                "style mix"
                inject_index = random.randint(1, self.n_latent - 1)

                w1 = w1.unsqueeze(1).repeat(1, inject_index, 1)
                w2 = w2.unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

                latent_w = torch.cat([w1, w2], 1)

            else:
                latent_z = torch.randn([self.batch_size, 512], device= self.device)
                latent_w = self.g_module.style(latent_z)

            "truncation trick"
            latent_w = self.w_mean + self.truncation * (latent_w - self.w_mean)


            # random_choice    = np.random.choice(self.truncation_latent.size(0),self.batch_size)
            # truncation_latent = self.truncation_latent[random_choice]

            # choice = [torch.abs(self.truncation_latent.reshape(self.truncation_latent.size(0),-1) - latent_.reshape(1,-1)).sum(dim = 1).argmin().item()
            #           for latent_ in latent]
            # truncation_latent = self.truncation_latent[choice]

            # if self.w_plus:
            #     latent = latent.unsqueeze(1).repeat(1, self.n_latent, 1)
                # inject_index = np.random.randint(low=0, high=self.n_latent)
                # truncation_latent = truncation_latent.unsqueeze(0).repeat(self.batch_size,1, 1)
                # mix_latent = truncation_latent[:,inject_index:, :] + self.truncation * (latent[:, inject_index:, :] - truncation_latent[:, inject_index:, :])


                # latent[:, inject_index:, :] = mix_latent

    
        return latent_w


class BrainCT(torch.utils.data.Dataset):
    # for projector.py
    def __init__(self, query_save_dir, transform, reverse=True):
        self.transform  = transform

        self.imgs      = sorted(glob(os.path.join(query_save_dir, 'png', '*.png')), reverse = reverse)[:32]
        self.fNames    = [img.split('/')[-1] for img in self.imgs]
        self.imgs      = [Image.open(img) for img in self.imgs]

        bet_npy        = os.path.join(query_save_dir, 'BrainExtraction', 'bet.npy')
        if os.path.exists(bet_npy): bet_np = np.load(bet_npy)
        else:                       bet_np = np.ones([len(self.imgs),512,512])
        if reverse:
            self.bet_masks = torch.from_numpy(np.flip(bet_np, 0).copy())[:32]
        else:
            self.bet_masks = torch.from_numpy(bet_np)[:32]

        target_npy     = os.path.join(query_save_dir, 'target.npy')
        if os.path.exists(target_npy): target_np = np.load(target_npy)
        else:                          target_np = np.zeros([len(self.imgs),512,512])

        if reverse:
            self.targets   = torch.from_numpy(np.flip(target_np, 0).copy())[:32]
        else:
            self.targets   = torch.from_numpy(target_np)[:32]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        fName, img, bet_mask, target = self.fNames[idx], self.imgs[idx], self.bet_masks[idx], self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return idx, fName, img, bet_mask, target

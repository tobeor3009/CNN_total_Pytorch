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
from torchvision.transforms import ToTensor
from PIL import Image

from .utils import *
from torchvision import utils
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pydicom
import json

class RealDataset(torch.utils.data.Dataset):
    def __init__(self,
                 split='train',
                 transform=None,
                 use_slice = False,
                 windowing = False,
                 urgency = ["normal"],
                 dcm_path = "../Dataset/Asan_Brain_CT/train/dicom"):
        self.split          = split
        self.transform      = transform
        self.urgency        = urgency
        dcms = {label: glob(os.path.join(dcm_path, label, "enrolled", "*", "*.dcm"))
                for label in self.urgency}

        self.imgs = []
        for label in self.urgency:
            self.imgs += imgs[label][:-10000] if self.split == "train" else imgs[label][-10000:]        
        random.shuffle(self.imgs)

        if use_slice:
            if self.split == "train":
                self.imgs = random.choices(self.imgs, k=100000)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
#         img = Image.open(self.imgs[idx])
        img = dcm2np(self.imgs[idx], windowing=True)

        if self.transform:
            img = self.transform(img)

        return img, label

    
class BrainDicom(torch.utils.data.Dataset):
    def __init__(self,
                 transform=None,
                 use_slice = False,
                 windowing = False,
                 urgency = ["normal"],
                 dcm_path = "../Dataset/Asan_Brain_CT/train/dicom"):
        self.transform      = transform
        self.windowing      = windowing

        self.patients = glob(os.path.join(dcm_path, urgency, "enrolled", "*"))

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        dcms = glob(os.path.join(self.patients[idx], "*.dcm"))
        img = dcm2np(random.choice(dcms), windowing=self.windowing)
            
        if self.transform:
            img = self.transform(img)
        return img


class DicomLoader(Dataset):
    def __init__(self, transform, path = "../Dataset/Asan_Brain_CT/train/dicom/", severities = ["normal", "benign"]):   
        self.patients   = {}
        self.n_patients = {}
        for severity in severities:
            self.patients[severity] = glob(os.path.join(path, severity, "enrolled", "*"))
        
    def __len__(self):
        n_patients = 0
        for severity in self.patients:
            n_patients += len(self.patients[severity])

        return n_patients
    
    def __getitem__(self, idx):
        n = 0
        for severity in self.patients:
            n_ = len(self.patients[severity])
            if n <= idx < n + n_:
                patient = self.patients[severity][idx - n]
                dcms = glob(os.path.join(patient, "*.dcm"))
                imgs = [dcm2np(dcm) for dcm in sorted(dcms)]
                patient_id, age, sex = get_patient_info(dcms[0])

                data = {"imgs"      : imgs,
                        "id"        : patient_id,
                        "age"       : age,
                        "severity"  : severity,
                        "sex"       : sex,
                        }
                break

            else:
                n += n_
                
        return imgs



class AxialScan(torch.utils.data.Dataset):
    def __init__(self,
                 split='train',
                 transform=None,
                 data_path = "../Dataset/Asan_Brain_CT/train/png/normal/windowed",
                 n_slices = 32,
                 reverse = True,
                 cond = True,
                 ):
        self.split          = split
        self.transform      = transform
        self.n_slices       = n_slices
        self.reverse        = reverse
        self.totensor       = ToTensor()

        self.query_dirs = [query_dir for query_dir in glob(os.path.join(data_path, "*")) if os.path.isdir(query_dir)]
        if cond:
            with open(os.path.join(data_path, "dataset.json"), "r") as f:
                dataset_json = json.load(f)
                self.labels = {data[0].split('/')[0]: data[1] for data in dataset_json["labels"]} 
                # data[0]: fname, data[1]: lables
                
        if self.split == "train":
            self.query_dirs = self.query_dirs[:-500]
        else:
            self.query_dirs = self.query_dirs[-500:]
        
    def __len__(self):
        return len(self.query_dirs)

    def __getitem__(self, idx):
        imgs = sorted(glob(os.path.join(self.query_dirs[idx], "*.png")), reverse = self.reverse)[:32]
        imgs = [self.totensor(Image.open(img)) for img in imgs]
        if self.transform:
            imgs = [self.transform(img) for img in imgs]

        if self.n_slices:
            if len(imgs) > self.n_slices:
                start_idx = random.randint(0, len(imgs) - self.n_slices)
                end_idx = start_idx + self.n_slices
                imgs = imgs[start_idx:end_idx]
            else:
                imgs = (self.n_slices - len(imgs)) * [imgs[0]] + imgs
        imgs = torch.stack(imgs).float()
        labels = torch.FloatTensor(self.labels[self.query_dirs[idx].split('/')[-1]])
        return imgs, labels
        # else:
        #     return self.__getitem__(random.randint(0, self.__len__()))

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


#         bet_npy        = os.path.join(query_save_dir, 'BrainExtraction', 'bet.npy')
        bet_npy        = os.path.join(query_save_dir, 'bet.npy')
        bet_np = np.load(bet_npy) if os.path.exists(bet_npy) else np.ones([len(self.imgs),512,512])
        if reverse: bet_np = np.flip(bet_np, 0).copy()
        self.bet_masks = torch.from_numpy(bet_np)[:32]

        target_npy     = os.path.join(query_save_dir, 'target.npy')
        if os.path.exists(target_npy): target_np = np.load(target_npy)
        else:                          target_np = np.zeros([len(self.imgs),512,512])
        if reverse: target_np = np.flip(target_np, 0).copy()
        self.targets   = torch.from_numpy(target_np)[:32]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        fName, img, bet_mask, target = self.fNames[idx], self.imgs[idx], self.bet_masks[idx], self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return idx, fName, img, bet_mask, target

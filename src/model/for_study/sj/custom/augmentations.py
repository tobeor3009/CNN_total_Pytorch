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
from torchvision import transforms
from PIL import Image

import lpips

from model import Generator
from model import Discriminator

from torchvision import utils

from custom.utils import mkdir

class ApplyOneof(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        transform = random.choice(self.transforms)
        return transform(img)

class Identity(object):
    def __call__(self, img):
        return img

class AddNoisePatch(object):
    def __init__(self, std=1, p = 1):
        self.std    = std
        self.p      = p
    
    def __call__(self, img): 
        if torch.rand(1) > self.p:
            return img
        else:
            b, c, h, w = img.size()
            patch = torch.zeros_like(img)
            (h_1, h_2), _ = (torch.rand(2) * h).int().sort()
            (w_1, w_2), _ = (torch.rand(2) * w).int().sort()
            patch[:, :, h_1:h_2, w_1:w_2] = 1

            mean  = (2 * torch.rand(b) - 1).reshape(b, 1, 1, 1).repeat(1, c, h ,w)
            noise = (2 * torch.rand(img.size()) - 1) * self.std + mean
            noise = noise.cuda()
            
            noise *= patch 
            return torch.clamp(img + noise, -1, 1)

    def __repr__(self):
        return self._class__.__name__ + f"(mean,std)=({mean},{std})"

class AddGaussianNoise(object):
    def __init__(self, mean = 0., std=1., p = 0.5):
        self.mean   = mean
        self.std    = std
        self.p      = p
    
    def __call__(self, img): 
        if torch.rand(1) > self.p:
            return img
        else:
            noise = (2  * torch.rand_like(img) - 1) * self.std + self.mean
            return torch.clamp(img + noise.cuda(), -1, 1)
    
    def __repr__(self):
        return self._class__.__name__ + f"(mean,std)=({mean},{std})"

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        min_output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, min_output_size=256 + 128, p=1, iou=1):
        assert isinstance(min_output_size, int)
        self.min_output_size = min_output_size
        self.p = p
        self.iou = iou

    def __call__(self, image, target, mask):
        if self.p < torch.rand(1):
            return image, target

        h, w = image.shape[:2]
        # image : H X W X C

        mask_sum = mask.sum()

        if target == -1:
            if mask_sum:
                self.min_output_size = int(mask_sum ** 0.5)
            new_size = torch.randint(low  = self.min_output_size, 
                                     high = 512)

            cnt = 0
            while True:
                cnt += 1
                if cnt == 10:
                    return image, target
                top = torch.randint(0, h - new_size)
                left = torch.randint(0, w - new_size)
                msk_sum = mask[top: top + new_size,
                          left: left + new_size].sum()
                if msk_sum >= self.iou * mask_sum:
                    image = image[top: top + new_size,
                            left: left + new_size]
                    return image, target
        else:
            new_size = torch.randint(self.min_output_size, 512)
            cnt = 0
            while True:
                cnt += 1
                top = torch.randint(0, h - new_size)
                left = torch.randint(0, w - new_size)

                img = image[top: top + new_size,
                      left: left + new_size].copy()
                if (img == 0).sum() < new_size ** 2:
                    return img, target

                if cnt == 10:
                    return image, target

        return image, target


def random_hflip(img, p = 0.5):
    if random.random() > p:
        return img
    else:
        return transforms.functional.hflip(img)

def random_rotate(img, max_angle = 45, p = 0.5):
    # img: tensor value in [-1, 1]
    # angle: rotate angle value in degrees, counter-clockwise
    if random.random() > p:
        return img
    else:
        random_angle = (2 * random.random() - 1) * max_angle # (-angle, angle]
        return transforms.functional.rotate(img + 1.0, random_angle) - 1.0

def random_aug_g(img):
    transforms = [random_hflip, random_rotate]
    if len(img.shape) == 5:
        b, n_slice, c, h, w = img.shape
        img = img.reshape(b * n_slice, c, h, w)
        for t in transforms:
            img = t(img)
        return img.reshape(b, n_slice, c, h, w)

    for t in transforms:
        img = t(img)
    return img

# base module
from tqdm import tqdm
import os
# external module
import torch
import numpy as np

# this library module
from .utils import imread, get_parent_dir_name
from .data_utils import get_resized_array, get_seg_augumented_array, get_preprocessed_array, base_augmentation_policy_dict
from .base_loader import BaseDataset


class SegDataset(BaseDataset):
    def __init__(self,
                 image_path_list=None,
                 mask_path_list=None,
                 imread_policy={"image": None, "mask": None},
                 on_memory=False,
                 augmentation_proba=False,
                 augmentation_policy_dict=base_augmentation_policy_dict,
                 image_channel_dict={"image": "rgb", "mask": "grayscale"},
                 preprocess_dict={"image": "-1~1", "mask": "0~1"},
                 target_size=None,
                 interpolation="bilinear",
                 dtype=torch.float32):
        super().__init__()

        self.image_path_list = [image_path for image_path in image_path_list]
        self.mask_path_list = [mask_path for mask_path in mask_path_list]

        self.imread_policy = imread_policy
        self.on_memory = on_memory
        self.is_data_ready = False if on_memory else True
        self.augmentation_proba = augmentation_proba
        self.augmentation_policy_dict = augmentation_policy_dict
        self.image_channel = image_channel_dict["image"]
        self.mask_channel = image_channel_dict["mask"]
        self.image_preprocess = preprocess_dict["image"]
        self.mask_preprocess = preprocess_dict["mask"]
        self.target_size = target_size
        self.interpolation = interpolation
        self.dtype = dtype

        if self.on_memory is True:
            self.data_on_memory_list = [None for _ in range(len(self))]
            self.get_data_on_ram()

        self.print_data_info()

    def __getitem__(self, i):

        if i >= len(self):
            raise IndexError

        current_index = i

        if self.on_memory and self.is_data_ready:
            image_array, mask_array = \
                self.data_on_memory_list[current_index]
        else:
            image_path = self.image_path_list[current_index]
            mask_path = self.mask_path_list[current_index]

            image_array = imread(image_path,
                                 policy=self.imread_policy["image"],
                                 channel=self.image_channel)
            mask_array = imread(mask_path,
                                policy=self.imread_policy["mask"],
                                channel=self.mask_channel)

            image_array = get_resized_array(image_array,
                                            self.target_size,
                                            self.interpolation)
            mask_array = get_resized_array(mask_array,
                                           self.target_size,
                                           "nearest")
        if (not self.on_memory) or (self.on_memory and self.is_data_ready):
            image_array, mask_array = get_seg_augumented_array(image_array, mask_array,
                                                               self.augmentation_proba,
                                                               self.augmentation_policy_dict)
            image_array = get_preprocessed_array(image_array,
                                                 self.image_preprocess)
            mask_array = get_preprocessed_array(mask_array,
                                                self.mask_preprocess)

            image_array = torch.as_tensor(image_array, dtype=self.dtype)
            mask_array = torch.as_tensor(mask_array, dtype=self.dtype)
        return image_array, mask_array

    def print_data_info(self):
        data_num = len(self)
        print(f"Total data num {data_num}")

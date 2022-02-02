# base module
from tqdm import tqdm
import os
# external module
import numpy as np

# this library module
from .utils import imread, get_parent_dir_name
from .data_utils import get_resized_array, get_augumented_array, get_preprocessed_array, base_augmentation_policy_dict
from .base_loader import BaseDataset

"""
Expected Data Path Structure

Example)
train - negative
      - positive
valid - negative
      - positive
test - negative
     - positive

"""


class ClassifyDataset(BaseDataset):

    def __init__(self,
                 image_path_list=None,
                 label_policy=None,
                 label_level=1,
                 on_memory=False,
                 argumentation_proba=False,
                 argumentation_policy_dict=base_augmentation_policy_dict,
                 image_channel_dict={"image": "rgb"},
                 preprocess_input="-1~1",
                 target_size=None,
                 interpolation="bilinear",
                 class_mode="binary",
                 dtype="float32"):
        super().__init__()

        self.image_path_list = [image_path for image_path in image_path_list]
        self.label_policy = label_policy
        self.label_level = label_level
        self.on_memory = on_memory
        self.is_data_ready = False if on_memory else True
        self.argumentation_proba = argumentation_proba
        self.argumentation_policy_dict = argumentation_policy_dict
        self.image_channel = image_channel_dict["image"]
        self.preprocess_input = preprocess_input
        self.target_size = target_size
        self.interpolation = interpolation
        self.class_mode = class_mode
        self.dtype = dtype

        self.is_class_cached = False
        self.class_list = [None for _ in range(len(self))]
        if self.on_memory is True:
            self.data_on_memory_list = [None for _ in range(len(self))]
            self.get_data_on_ram()

        self.print_data_info()

    def __getitem__(self, i):

        if i >= len(self):
            raise IndexError

        current_index = i

        if self.on_memory and self.is_data_ready:
            image_array, label = \
                self.data_on_memory_list[current_index]
            image_array = get_augumented_array(image_array,
                                               self.argumentation_proba,
                                               self.argumentation_policy_dict)
            image_array = get_preprocessed_array(image_array,
                                                 self.preprocess_input)
        else:
            image_path = self.image_path_list[current_index]
            image_array = imread(image_path, channel=self.image_channel)
            image_array = get_resized_array(image_array,
                                            self.target_size,
                                            self.interpolation)
            if not self.on_memory:
                image_array = get_augumented_array(image_array,
                                                   self.argumentation_proba,
                                                   self.argumentation_policy_dict)
                image_array = get_preprocessed_array(image_array,
                                                     self.preprocess_input)
            if self.is_class_cached:
                label = self.class_list[current_index]
            else:
                image_dir_name = get_parent_dir_name(image_path,
                                                     self.label_level)
                label = self.label_policy(image_dir_name)
                self.class_list[current_index] = label
                self.is_class_cached = self.check_class_list_cached()

        image_array = image_array.astype(self.dtype)
        # label = label.astype(self.dtype)
        return image_array, label

    def print_data_info(self):
        data_num = len(self)
        print(f"Total data num {data_num}")

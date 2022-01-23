# base module
from tqdm import tqdm
import os
# external module
import numpy as np

# this library module
from .utils import imread, get_parent_dir_name
from .data_utils import get_resized_array, get_augumented_array, get_preprocessed_array, base_argumentation_policy_dict
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
                 label_to_index_dict=None,
                 label_level=1,
                 on_memory=False,
                 argumentation_proba=False,
                 argumentation_policy_dict=base_argumentation_policy_dict,
                 image_channel_dict={"image": "rgb"},
                 image_data_format="channel_first",
                 preprocess_input="-1~1",
                 target_size=None,
                 interpolation="bilinear",
                 class_mode="binary",
                 dtype="float32"):
        super().__init__()

        self.image_path_dict = {index: image_path for index,
                                image_path in enumerate(image_path_list)}
        self.data_on_memory_dict = {}
        self.label_to_index_dict = label_to_index_dict
        self.label_level = label_level
        self.num_classes = len(self.label_to_index_dict)
        self.on_memory = on_memory
        self.is_data_ready = False if on_memory else True
        self.argumentation_proba = argumentation_proba
        self.argumentation_policy_dict = argumentation_policy_dict
        self.image_channel = image_channel_dict["image"]
        self.image_data_format = image_data_format
        self.preprocess_input = preprocess_input
        self.target_size = target_size
        self.interpolation = interpolation
        self.class_mode = class_mode

        self.is_class_cached = False
        self.class_dict = {i: None for i in range(len(self))}

        if self.on_memory is True:
            self.get_data_on_ram()

        self.print_data_info()

    def __getitem__(self, i):

        if i >= len(self):
            raise IndexError

        current_index = i

        if self.on_memory and self.is_data_ready:
            image_array, label = \
                self.data_on_memory_dict[current_index]
            image_array = get_augumented_array(image_array,
                                               self.argumentation_proba,
                                               self.argumentation_policy_dict)
            image_array = get_preprocessed_array(image_array,
                                                 self.preprocess_input)
        else:
            image_path = self.image_path_dict[current_index]
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
                label = self.class_dict[current_index]
            else:
                image_dir_name = get_parent_dir_name(image_path,
                                                     self.label_level)
                label = self.label_to_index_dict[image_dir_name]
                self.class_dict[current_index] = label
                self.is_class_cached = self.check_class_dict_cached()

        if self.image_data_format == "channel_first" and self.is_data_ready:
            image_array = np.rollaxis(image_array, 2, 0)
        return image_array, label

    def print_data_info(self):
        data_num = len(self)
        print(f"Total data num {data_num} with {self.num_classes} classes")

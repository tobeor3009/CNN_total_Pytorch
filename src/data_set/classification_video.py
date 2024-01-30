# base module
from glob import glob
from typing import Any
# external module
import numpy as np
import random

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

from vidaug import augmentors as va


# Used to apply augmentor with 50% probability
def sometimes(aug): return va.Sometimes(0.9, aug)


blur_aug = va.OneOf([
    sometimes(va.GaussianBlur(sigma=1)),
    sometimes(va.Pepper()),
    sometimes(va.Salt())
])

video_aug_seq = va.OneOf([
    blur_aug,
    # sometimes(va.InvertColor()),
    sometimes(va.VerticalFlip()),
    sometimes(va.HorizontalFlip())
])


def get_aug_video_array(video_array):
    aug_video_array = video_aug_seq(video_array)
    aug_video_array = np.stack(aug_video_array, axis=0)
    return aug_video_array


class ClassifyDataset(BaseDataset):

    def __init__(self,
                 image_folder_list: list = [],
                 label_policy: list = [],
                 label_level: list = 1,
                 on_memory: bool = False,
                 augmentation_proba: bool = False,
                 augmentation_policy_dict: dict = base_augmentation_policy_dict,
                 image_channel_dict: dict = {"image": "rgb"},
                 preprocess_input: Any = "-1~1",
                 target_size: tuple = None,
                 interpolation: str = "bilinear",
                 class_mode: str = "binary",
                 dtype: torch.dtype = torch.float32):
        super().__init__()

        self.image_folder_list = [
            image_folder for image_folder in image_folder_list]
        self.label_policy = label_policy
        self.label_level = label_level
        self.on_memory = on_memory
        self.is_data_ready = False if on_memory else True
        self.augmentation_proba = augmentation_proba
        self.augmentation_policy_dict = augmentation_policy_dict
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
            video_array, label = \
                self.data_on_memory_list[current_index]
            video_array = get_aug_video_array(video_array)
            video_array = get_preprocessed_array(video_array,
                                                 self.preprocess_input)
        else:
            image_folder = self.image_folder_list[current_index]
            image_path_list = sorted(glob(f"{image_folder}/*.png"))
            video_array = []
            for image_path in image_path_list:
                image_array = imread(image_path, channel=self.image_channel)
                image_array = get_resized_array(image_array,
                                                self.target_size,
                                                self.interpolation)
                video_array.append(image_array)
            video_array = np.stack(video_array, axis=0)
            if not self.on_memory:
                video_array = get_aug_video_array(video_array)
                video_array = get_preprocessed_array(video_array,
                                                     self.preprocess_input)
            if self.is_class_cached:
                label = self.class_list[current_index]
            else:
                image_dir_name = get_parent_dir_name(image_folder,
                                                     self.label_level)
                label = self.label_policy(image_dir_name)
                self.class_list[current_index] = label
                self.is_class_cached = self.check_class_list_cached()

        video_array = torch.as_tensor(video_array, dtype=self.dtype)
        label = torch.as_tensor(label, dtype=self.dtype)
        return video_array, label

    def __len__(self):
        return len(self.image_folder_list)

    def print_data_info(self):
        data_num = len(self)
        print(f"Total data num {data_num}")


class AgeGenderClassifyDataset(BaseDataset):

    def __init__(self,
                 image_path_list=None,
                 label_policy=None,
                 iter_len=10000,
                 label_level=1,
                 on_memory=False,
                 augmentation_proba=False,
                 augmentation_policy_dict=base_augmentation_policy_dict,
                 image_channel_dict={"image": "rgb"},
                 preprocess_input="-1~1",
                 target_size=None,
                 interpolation="bilinear",
                 class_mode="binary",
                 dtype=torch.float32):
        super().__init__()

        image_path_list = [image_path for image_path in image_path_list]
        self.image_path_dict = {}
        for image_path in image_path_list:
            age = get_parent_dir_name(image_path, level=1).split("_")[0]
            if age in self.image_path_dict:
                self.image_path_dict[age].append(image_path)
            else:
                self.image_path_dict[age] = [image_path]
        self.age_keys = list(self.image_path_dict.keys())

        self.iter_len = iter_len
        self.label_policy = label_policy
        self.label_level = label_level
        self.on_memory = on_memory
        self.is_data_ready = False if on_memory else True
        self.augmentation_proba = augmentation_proba
        self.augmentation_policy_dict = augmentation_policy_dict
        self.image_channel = image_channel_dict["image"]
        self.preprocess_input = preprocess_input
        self.target_size = target_size
        self.interpolation = interpolation
        self.class_mode = class_mode
        self.dtype = dtype

        if self.on_memory is True:
            self.data_on_memory_list = [None for _ in range(len(self))]
            self.get_data_on_ram()

        self.print_data_info()

    def __getitem__(self, i):

        if i >= len(self):
            raise IndexError

        current_index = i

        age = random.choice(self.age_keys)
        image_path = random.choice(self.image_path_dict[age])

        image_array = imread(image_path, channel=self.image_channel)
        image_array = get_resized_array(image_array,
                                        self.target_size,
                                        self.interpolation)
        image_array = get_augumented_array(image_array,
                                           self.augmentation_proba,
                                           self.augmentation_policy_dict)
        image_array = get_preprocessed_array(image_array,
                                             self.preprocess_input)
        image_dir_name = get_parent_dir_name(image_path,
                                             self.label_level)
        label = self.label_policy(image_dir_name)

        image_array = torch.as_tensor(image_array, dtype=self.dtype)
        label = torch.as_tensor(label, dtype=self.dtype)
        # image_array = image_array.astype(self.dtype)
        # label = label.astype(self.dtype)
        return image_array, label

    def print_data_info(self):
        data_num = len(self)
        print(f"Total data num {data_num}")

    def __len__(self):
        return self.iter_len


class AgeGenderClassifyDataset(BaseDataset):

    def __init__(self,
                 image_path_list=None,
                 label_policy=None,
                 label_level=1,
                 num_data_each_age=None,
                 on_memory=False,
                 augmentation_proba=False,
                 augmentation_policy_dict=base_augmentation_policy_dict,
                 image_channel_dict={"image": "rgb"},
                 preprocess_input="-1~1",
                 target_size=None,
                 interpolation="bilinear",
                 class_mode="binary",
                 dtype=torch.float32):
        super().__init__()

        self.image_path_list = [image_path for image_path in image_path_list]
        self.image_num = len(image_path_list)
        self.image_info_dict = {}
        self.inner_index = 0
        for image_path in image_path_list:
            age_gender = get_parent_dir_name(image_path, level=1)
            if age_gender in self.image_info_dict:
                if self.image_info_dict[age_gender] > num_data_each_age - 1:
                    continue
                self.image_info_dict[age_gender] += 1
            else:
                # age_gender = "0.95_0"
                self.image_info_dict[age_gender] = 1
        self.current_image_info_dict = None
        self.reset_current_image_info_dict()
        self.iter_len = np.sum(list(self.image_info_dict.values()))
        self.label_policy = label_policy
        self.label_level = label_level
        self.on_memory = on_memory
        self.is_data_ready = False if on_memory else True
        self.augmentation_proba = augmentation_proba
        self.augmentation_policy_dict = augmentation_policy_dict
        self.image_channel = image_channel_dict["image"]
        self.preprocess_input = preprocess_input
        self.target_size = target_size
        self.interpolation = interpolation
        self.class_mode = class_mode
        self.dtype = dtype

        if self.on_memory is True:
            self.data_on_memory_list = [None for _ in range(len(self))]
            self.get_data_on_ram()

        self.print_data_info()

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError

        image_path = self.get_next_image_path()
        image_array = imread(image_path, channel=self.image_channel)
        image_array = get_resized_array(image_array,
                                        self.target_size,
                                        self.interpolation)
        image_array = get_augumented_array(image_array,
                                           self.augmentation_proba,
                                           self.augmentation_policy_dict)
        image_array = get_preprocessed_array(image_array,
                                             self.preprocess_input)
        image_dir_name = get_parent_dir_name(image_path,
                                             self.label_level)
        label = self.label_policy(image_dir_name)
        image_array = torch.as_tensor(image_array, dtype=self.dtype)
        label = torch.as_tensor(label, dtype=self.dtype)

        return image_array, label

    def print_data_info(self):
        data_num = len(self)
        print(f"Total data num {data_num}")

    def __len__(self):
        return self.iter_len

    def reset_current_image_info_dict(self):
        self.current_image_info_dict = {
            key: 0 for key in self.image_info_dict.keys()}
        random.shuffle(self.image_path_list)
        self.inner_index = 0

    def get_next_image_path(self):
        data_ready = False
        image_path = None
        while not data_ready:
            if self.inner_index == self.image_num:
                self.reset_current_image_info_dict()
            image_path = self.image_path_list[self.inner_index]
            age_gender = get_parent_dir_name(image_path, level=1)
            total_data_num = self.image_info_dict[age_gender]
            current_data_num = self.current_image_info_dict[age_gender]

            if current_data_num >= total_data_num:
                pass
            else:
                self.current_image_info_dict[age_gender] += 1
                data_ready = True
            self.inner_index += 1
        return image_path

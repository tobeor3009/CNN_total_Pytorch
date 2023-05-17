from abc import abstractmethod

import torch
import torch.utils.data as data

import cv2
import progressbar
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle as syncron_shuffle
import albumentations as A


class BaseDataset(data.Dataset):

    def __init__(self):
        self.image_path_list = None
        self.data_on_memory_list = None
        self.on_memory = False
        self.data_len = None
        self.data_index_list = None

    def __len__(self):
        if self.data_len is None:
            self.data_len = len(self.image_path_list)

        return self.data_len

    def get_data_on_ram(self):
        widgets = [
            ' [',
            progressbar.Counter(format=f'%(value)02d/%(max_value)d'),
            '] ',
            progressbar.Bar(),
            ' (',
            progressbar.ETA(),
            ') ',
        ]
        progressbar_displayed = progressbar.ProgressBar(widgets=widgets,
                                                        maxval=len(self)).start()
        for index, data_tuple in enumerate(self):
            self.data_on_memory_list[index] = data_tuple
            progressbar_displayed.update(value=index + 1)
        self.is_data_ready = True
        progressbar_displayed.finish()

    def check_class_list_cached(self):
        for value in self.class_list:
            if value is None:
                return False
        return True


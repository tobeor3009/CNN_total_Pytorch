import torch.utils.data as data

import numpy as np


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
    
    def check_class_list_cached(self):
        for value in self.class_list:
            if value is None:
                return False
        return True

class DatasetCache(data.Dataset):
    def __init__(self, dataset, preloading=True):
        self.dataset = dataset
        self.preloading = preloading
        if self.preloading:
            self.data_list = []
            self.preloading_data()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.preloading:
            return self.lazy_load(idx)
        else:
            return self.memory_load(idx)

    def preloading_data(self):
        for idx in range(len(self)):
            data = self.lazy_load(idx)
            self.data_list.append(data)

    def memory_load(self, idx):
        return self.data_list[idx]

    def lazy_load(self, idx):
        data = self.dataset[idx]
        return data

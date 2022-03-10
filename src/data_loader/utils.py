import cv2
import os
import numpy as np
from collections.abc import Mapping


def imread(img_path, channel=None):
    extension = os.path.splitext(img_path)[1]
    if extension == ".npy":
        img_numpy_array = np.load(
            img_path, allow_pickle=True).astype("float32")
    else:
        img_byte_stream = open(img_path.encode("utf-8"), "rb")
        img_byte_array = bytearray(img_byte_stream.read())
        img_numpy_array = np.asarray(img_byte_array, dtype=np.uint8)

        if channel == "rgb":
            img_numpy_array = cv2.imdecode(
                img_numpy_array, cv2.IMREAD_UNCHANGED)
            img_numpy_array = cv2.cvtColor(
                img_numpy_array, cv2.COLOR_BGR2RGB)
        elif channel == "grayscale":
            img_numpy_array = cv2.imdecode(
                img_numpy_array, cv2.IMREAD_GRAYSCALE)
        else:
            img_numpy_array = cv2.imdecode(
                img_numpy_array, cv2.IMREAD_UNCHANGED)
    return img_numpy_array


def get_parent_dir_name(path, level=1):

    path_spliter = os.path.sep
    abs_path = os.path.abspath(path)

    return abs_path.split(path_spliter)[-(1 + level)]


class LazyDict(Mapping):
    def __init__(self, *args, **kw):
        self._raw_dict = dict(*args, **kw)

    def __getitem__(self, key):
        func, arg = self._raw_dict.__getitem__(key)
        return func(arg)

    def __iter__(self):
        return iter(self._raw_dict)

    def __len__(self):
        return len(self._raw_dict)


def get_array_dict_lazy(key_tuple, array_tuple):
    return lambda index: {key: array[index] for key, array in zip(key_tuple, array_tuple)}


def get_npy_array(path, target_size, data_key, shape, dtype):

    path_spliter = os.path.sep
    abs_path = os.path.abspath(path)
    data_sort_list = ["train", "valid", "test"]

    splited_path = abs_path.split(path_spliter)
    for index, folder in enumerate(splited_path):
        if folder == "datasets":
            break

    for folder in splited_path:
        find_data_sort = False

        for data_sort in data_sort_list:
            if folder == data_sort:
                find_data_sort = True
                break
        if find_data_sort is True:
            break
    # index mean datasets folder. so it means ~/datasets/task/name
    current_data_folder = path_spliter.join(splited_path[:index + 3])

    common_path = f"{current_data_folder}/{data_sort}_{target_size}_{data_key}"

    memmap_npy_path = f"{common_path}.npy"
    lock_path = f"{common_path}.lock"

    if os.path.exists(lock_path):
        memmap_array = np.memmap(
            memmap_npy_path, dtype=dtype, mode="r", shape=shape)
    else:
        memmap_array = \
            np.memmap(memmap_npy_path, dtype=dtype, mode="w+", shape=shape)

    return memmap_array, lock_path

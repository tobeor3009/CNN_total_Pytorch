import torch
from time import sleep
import multiprocessing as mp
from multiprocessing.queues import Empty
import math
import random
import numpy as np

WAIT_TIME = 0.05


def default_collate_fn(data_object_list):

    batch_image_array = []
    batch_label_array = []
    for image_array, label_array in data_object_list:
        batch_image_array.append(image_array)
        batch_label_array.append(label_array)
    batch_image_array = torch.stack(batch_image_array, dim=0)
    batch_label_array = torch.stack(batch_label_array, dim=0)

    return batch_image_array, batch_label_array


def consumer_fn(dataset, idx_queue, output_queue, batch_size):
    inter_idx = 0
    while True:
        inter_idx += 1
        try:
            if output_queue.qsize() > batch_size * 10:
                sleep(WAIT_TIME)
                continue
            idx = idx_queue.get(timeout=0)
        except Empty:
            sleep(WAIT_TIME)
            continue
        data = (idx, dataset[idx])
        output_queue.put(data)


class BaseProcessPool():

    def shuffle_idx(self):
        if self.shuffle:
            random.shuffle(self.idx_list)


class SingleProcessPool(BaseProcessPool):
    def __init__(self, dataset, batch_size,
                 collate_fn=default_collate_fn, shuffle=False):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.shuffle = shuffle
        self.data_num = len(dataset)
        self.idx_list = list(range(self.data_num))
        self.batch_num = math.ceil(self.data_num / batch_size)
        self.batch_size = batch_size

        self.batch_idx = 0
        self.shuffle_idx()

    def __iter__(self):
        while self.batch_idx < self.batch_num:
            start_idx = self.batch_size * self.batch_idx
            end_idx = min(start_idx + self.batch_size, self.data_num)
            batch_idx_list = self.idx_list[start_idx: end_idx]

            data_object_list = []
            for batch_idx in batch_idx_list:
                data_object = self.dataset[batch_idx]
                data_object_list.append(data_object)

            batch_data_tuple = self.collate_fn(data_object_list)
            self.batch_idx += 1
            yield batch_data_tuple
        self.shuffle_idx()

    def __len__(self):
        return self.batch_num


class MultiProcessPool(BaseProcessPool):
    def __init__(self, dataset, batch_size, num_workers,
                 collate_fn=default_collate_fn, shuffle=False, verbose=False):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.shuffle = shuffle
        self.data_num = len(dataset)
        self.idx_list = list(range(self.data_num))
        self.process_list = None
        self.batch_num = math.ceil(self.data_num / batch_size)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.verbose = verbose
        self.idx_queue = None
        self.output_queue = None
        self.process_list = None

    def __iter__(self):
        self.start_process()
        try:
            while self.batch_idx < self.batch_num:
                start_idx = self.batch_size * self.batch_idx
                end_idx = min(start_idx + self.batch_size, self.data_num)
                current_batch_size = end_idx - start_idx
                data_object_list = []
                for _ in range(current_batch_size):
                    while len(data_object_list) < current_batch_size:
                        try:
                            batch_idx, data_object = self.output_queue.get(
                                timeout=0)
                            data_object_list.append(data_object)
                            if self.verbose:
                                print(batch_idx)
                        except Empty:
                            sleep(WAIT_TIME)
                            continue

                batch_data_tuple = self.collate_fn(data_object_list)

                self.batch_idx += 1
                yield batch_data_tuple
        finally:
            self.terminate_process()

    def put_idx_to_queue(self):
        for idx in self.idx_list:
            self.idx_queue.put(idx)

    def reset_states(self):
        self.batch_idx = 0
        self.shuffle_idx()
        self.put_idx_to_queue()

    def start_process(self):
        self.idx_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.reset_states()
        self.process_list = []
        for _ in range(self.num_workers):
            process = mp.Process(target=consumer_fn, args=(self.dataset,
                                                           self.idx_queue,
                                                           self.output_queue,
                                                           self.batch_size))
            # process.daemon = True
            process.start()
            self.process_list.append(process)
        if self.verbose:
            print("process_start")

    def terminate_process(self):
        if self.process_list is not None:
            for process in self.process_list:
                process.terminate()
                process.join()
            del self.process_list
            self.process_list = []
            self.idx_queue.close()
            self.output_queue.close()
            if self.verbose:
                print("process_terminate")

    def __len__(self):
        return self.batch_num


class DataLoader():
    def __init__(self, dataset, batch_size, collate_fn=default_collate_fn,
                 num_workers=0, pin_memory=True, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.data_num = len(self.dataset)
        self.batch_num = math.ceil(self.data_num / batch_size)
        self.idx_list = list(range(self.data_num))
        if num_workers <= 1:
            self.data_pool = SingleProcessPool(dataset, batch_size,
                                               collate_fn=collate_fn, shuffle=shuffle)
        else:
            self.data_pool = MultiProcessPool(dataset, batch_size, num_workers,
                                              collate_fn=collate_fn, shuffle=shuffle,
                                              verbose=False)

    def __iter__(self):
        iter_pool = iter(self.data_pool)
        for batch_data in iter_pool:
            batch_data = self.apply_pin_memory(batch_data)
            yield batch_data

    def __getitem__(self, i):
        start = i * self.batch_size
        end = min(start + self.batch_size, self.data_num)
        batch_idx_list = self.idx_list[start:end]
        batch_data = [self.dataset[data_idx]
                      for data_idx in batch_idx_list]
        batch_data = self.collate_fn(batch_data)
        batch_data = self.apply_pin_memory(batch_data)
        return batch_data

    def __len__(self):
        return self.batch_num

    def apply_pin_memory(self, batch_data):
        if self.pin_memory:
            batch_data = [item.pin_memory()
                          for item in batch_data]
        return batch_data

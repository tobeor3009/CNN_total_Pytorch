from tqdm import tqdm
import multiprocessing as mp

def fn_each_item(item, single_core_fn, index_term):
    total_idx, loop_item = item
    single_core_fn(loop_item)
    
    if index_term is not None:
        if int(total_idx) % index_term == 0:
            print(total_idx)


def threaded_process_fn(items_chunk, single_core_fn, use_tqdm, index_term):
    """ Your main process which runs in thread for each chunk"""
    if use_tqdm:
        items_chunk = tqdm(items_chunk, total=len(items_chunk))
    for item in items_chunk:
        fn_each_item(item, single_core_fn, index_term)


def do_multi_processing(item_list, single_core_fn, n_process, use_tqdm=False, index_term=1000):
    if use_tqdm:-
        index_term = None
    
    item_list = [(idx, item) for idx, item in enumerate(item_list)]
    array_chunk = np.array_split(item_list, n_process)
    for array in array_chunk:
        print(array[0][0], array[-1][0])
    process_list = []
    for process_idx in range(n_process):
        process = mp.Process(target=threaded_process_fn, 
                             args=(array_chunk[process_idx], single_core_fn, use_tqdm, index_term),)
        process_list.append(process)
        process_list[process_idx].start()
    for process in process_list:
        process.join()
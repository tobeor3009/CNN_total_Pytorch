import math
import collections
from itertools import repeat
from functools import partial
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def get_list_numel(list_obj):
    return int(np.prod(list_obj))

def get_array_numel(array_obj):
    return int(array_obj.prod())

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

def to_ntuple(value, n):
    return _ntuple(n)(value)

def assert_patch_stride_size(patch_size, stride):
    assert patch_size % stride == 0, "patch_size must be divisible by stride"

# expected shape = [B * num_patch, C, H // patch_size, W // patch_size]
def unfold_nd(input_tensor, patch_size, stride, pad_size):
    img_dim = input_tensor.dim() - 2
    unfold_dim_range = range(2, 2 + img_dim)
    pad_tuple = to_ntuple(pad_size, img_dim * 2)
    input_tensor = F.pad(input_tensor, pad_tuple, mode="constant", value=0)
    for unfold_dim in unfold_dim_range:
        input_tensor = input_tensor.unfold(unfold_dim, patch_size, stride)
    return input_tensor

def get_fold_idx_tensor(batch_size, in_channels, output_size, patch_size, stride, pad_size, img_dim, device):
    padded_output_size = output_size + pad_size * 2
    padded_output_size_tuple = to_ntuple(padded_output_size, img_dim)
    padded_output_size_numel = get_list_numel(padded_output_size_tuple)

    patch_per_dim = int((padded_output_size - patch_size) / stride) + 1
    patch_size_tuple = to_ntuple(patch_size, img_dim)
    patch_per_dim_tuple = to_ntuple(patch_per_dim, img_dim)
    num_patch = get_list_numel(patch_per_dim_tuple)
    patch_size_numel = get_list_numel(patch_size_tuple)

    fold_idx = torch.arange(padded_output_size_numel,
                            dtype=torch.int64, device=device).view(1, 1, *padded_output_size_tuple)
    fold_idx = unfold_nd(fold_idx, patch_size, stride, 0, img_dim)
    fold_idx = fold_idx.contiguous().view(1, num_patch, patch_size_numel).permute(0, 2, 1)
    fold_idx = fold_idx.contiguous().view(1, 1, -1).long().expand(batch_size, in_channels, -1)
    return fold_idx

# fold_nd input_shape: [B * patch_num, C, patch_size, patch_size]
def fold_nd(unfold_tensor, batch_size, output_size, patch_size, stride, pad_size, img_dim, fold_idx=None):

    padded_output_size = output_size + pad_size * 2
    padded_output_size_tuple = to_ntuple(padded_output_size, img_dim)
    padded_output_size_numel = get_list_numel(padded_output_size_tuple)

    patch_per_dim = int((padded_output_size - patch_size) / stride) + 1
    patch_size_tuple = to_ntuple(patch_size, img_dim)
    patch_per_dim_tuple = to_ntuple(patch_per_dim, img_dim)
    num_patch = get_list_numel(patch_per_dim_tuple)
    patch_size_numel = get_list_numel(patch_size_tuple)

    _, in_channels, *_ = unfold_tensor.shape

    if fold_idx is None:
        fold_idx = get_fold_idx_tensor(batch_size, in_channels, output_size, patch_size, stride, pad_size,
                                       img_dim, unfold_tensor.device)
    unfold_tensor = unfold_tensor.view(batch_size, num_patch, in_channels, patch_size_numel)
    unfold_tensor = unfold_tensor.permute(0, 2, 3, 1)
    unfold_tensor = unfold_tensor.contiguous().view(batch_size, in_channels, -1)

    output_tensor = torch.zeros(
        batch_size,
        in_channels,
        padded_output_size_numel,
        device=unfold_tensor.device,
        dtype=unfold_tensor.dtype,
    )
    output_tensor.scatter_add_(2, fold_idx, unfold_tensor)
    output_tensor = output_tensor.reshape(batch_size, in_channels, *padded_output_size_tuple)

    for narrow_dim in range(2, 2 + img_dim):
        output_tensor = output_tensor.narrow(narrow_dim, pad_size, output_size)

    return output_tensor

def extract_patch_tensor(input_tensor, patch_size, stride, pad_size):
    assert_patch_stride_size(patch_size, stride)
    _, C, *img_dim_list = input_tensor.shape
    patch_size_gen = [patch_size for _ in img_dim_list]
    img_dim = len(img_dim_list)
    if img_dim == 2:
        permute_dim = (0, 2, 3, 1, 4, 5)
    else:
        permute_dim = (0, 2, 3, 4, 1, 5, 6, 7)

    patch_tensor = unfold_nd(input_tensor, patch_size, stride, pad_size, img_dim)
    patch_tensor = patch_tensor.permute(*permute_dim)
    patch_tensor = patch_tensor.contiguous().view(-1, C, *patch_size_gen)
    return patch_tensor

def combine_region_voting_patches(unfold_tensor, batch_size, image_size, patch_size, stride, pad_size, img_dim):
    assert_patch_stride_size(patch_size, stride)
    _, in_channels, *_ = unfold_tensor.shape

    unfold_idx_tensor = torch.ones_like(unfold_tensor)
    fold_idx = get_fold_idx_tensor(batch_size, in_channels, image_size, patch_size, stride, pad_size, img_dim, unfold_tensor.device)
    recon_tensor = fold_nd(unfold_tensor, batch_size, image_size, patch_size, stride, pad_size, img_dim, fold_idx=fold_idx)
    divisor = fold_nd(unfold_idx_tensor, batch_size, image_size, patch_size, stride, pad_size, img_dim, fold_idx=fold_idx)
    
    return recon_tensor / divisor

def combine_region_voting_patches_with_patch_weights(unfold_tensor, batch_size, image_size, patch_size, stride, pad_size, img_dim,
                                                     stride_weight=None):
    assert_patch_stride_size(patch_size, stride)
    _, in_channels, *_ = unfold_tensor.shape
    if img_dim == 2:
        weight_permute_dim = (2, 0, 1)
    else:
        weight_permute_dim = (3, 0, 1, 2)
    expand_dim = to_ntuple(patch_size, img_dim)
    padded_image_size = image_size + pad_size * 2
    patch_per_dim = int((padded_image_size - patch_size) / stride) + 1
    num_patch = patch_per_dim ** img_dim
    device, dtype = unfold_tensor.device, unfold_tensor.dtype

    if stride_weight is None:
        stride_weight = compute_stride_weights(image_size, patch_size, stride, pad_size, img_dim).to(device=device, dtype=dtype)
    stride_weight = stride_weight.permute(*weight_permute_dim).unsqueeze(0).unsqueeze(2)
    stride_weight = stride_weight.expand(batch_size, num_patch, in_channels, *expand_dim).flatten(0, 1)

    fold_idx = get_fold_idx_tensor(batch_size, in_channels, image_size, patch_size, stride, pad_size, img_dim, unfold_tensor.device)
    recon_tensor = fold_nd(unfold_tensor, batch_size, image_size, patch_size, stride, pad_size, img_dim, fold_idx=fold_idx)
    divisor = fold_nd(stride_weight, batch_size, image_size, patch_size, stride, pad_size, img_dim, fold_idx=fold_idx)
    
    return recon_tensor / divisor

def compute_single_weights(patch_size, img_dim):
    patch_size_tuple = to_ntuple(patch_size, img_dim)
    if img_dim == 2:
        height, width = patch_size_tuple
        center_y, center_x = height // 2, width // 2
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    elif img_dim == 3:
        depth, height, width = patch_size_tuple
        center_z, center_y, center_x = depth // 2, height // 2, width // 2
        z_coords, y_coords, x_coords = np.indices(patch_size_tuple)
        distances = np.sqrt((z_coords - center_z) ** 2 + (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    
    max_distance = (patch_size / 2) ** (img_dim * 1.25)
    distances = 1 - (distances / max_distance).clip(0, 1)
    return torch.tensor(distances, dtype=torch.float32)

def compute_stride_weights(image_size, patch_size, stride, pad_size, img_dim):
    padded_image_size = image_size + pad_size * 2
    patch_per_dim = int((padded_image_size - patch_size) / stride) + 1
    num_patch = patch_per_dim ** img_dim
    repeat_dim = to_ntuple(1, img_dim)
    single_weight = compute_single_weights(patch_size, img_dim)[..., None]
    patch_weight = single_weight.repeat(*repeat_dim, num_patch)
    if img_dim == 2:
        patch_weight[:stride, :, :patch_per_dim] = 1
        patch_weight[:, :stride, range(0, num_patch, patch_per_dim)] = 1
        patch_weight[:, -stride:, range(patch_per_dim - 1, num_patch, patch_per_dim)] = 1
        patch_weight[-stride:, :, -patch_per_dim:] = 1
    else:
        block_jump_step = patch_per_dim ** 2
        patch_weight[:stride, :, :, :patch_per_dim] = 1
        patch_weight[:, :stride, :, range(0, num_patch, block_jump_step)] = 1
        patch_weight[:, -stride:, :, range(patch_per_dim - 1, num_patch, block_jump_step)] = 1
        patch_weight[:, :, :stride, range(0, num_patch, block_jump_step)] = 1
        patch_weight[:, :, -stride:, range(patch_per_dim - 1, num_patch, block_jump_step)] = 1
        patch_weight[-stride:, :, :, -patch_per_dim:] = 1
    return patch_weight

def process_patch_array(patch_tensor, target_model, process_at_once, part_process_fn=None, dynamic=False):
    if dynamic:
        return _process_patch_array_dynamic(patch_tensor, target_model, process_at_once, part_process_fn=part_process_fn)
    else:
        return _process_patch_array_normal(patch_tensor, target_model, process_at_once, part_process_fn=part_process_fn)

def _process_patch_array_dynamic(patch_tensor, target_model, process_at_once, part_process_fn=None):
    data_num = patch_tensor.shape[0]
    device = next(target_model.parameters()).device
    batch_num = math.ceil(data_num / process_at_once)
    pred_patch_array = []
    for batch_idx in range(batch_num):
        start_idx = batch_idx * process_at_once
        end_idx = min(start_idx + process_at_once, data_num)
        target_data = patch_tensor[start_idx:end_idx]
        target_data = target_data.to(device)
        pred_patch_array_part = target_model(target_data)
        if part_process_fn is not None:
            pred_patch_array_part = part_process_fn(pred_patch_array_part)
        pred_patch_array_part = pred_patch_array_part.cpu()
        pred_patch_array.append(pred_patch_array_part)
    pred_patch_array = torch.cat(pred_patch_array, axis=0)
    return pred_patch_array

def _process_patch_array_normal(patch_tensor, target_model, process_at_once, part_process_fn=None):
    data_num = patch_tensor.shape[0]
    batch_num = math.ceil(data_num / process_at_once)
    pred_patch_array = []
    for batch_idx in range(batch_num):
        start_idx = batch_idx * process_at_once
        end_idx = min(start_idx + process_at_once, data_num)
        pred_patch_array_part = target_model(patch_tensor[start_idx:end_idx])
        if part_process_fn is not None:
            pred_patch_array_part = part_process_fn(pred_patch_array_part)
        pred_patch_array.append(pred_patch_array_part)
    pred_patch_array = torch.cat(pred_patch_array, axis=0)
    return pred_patch_array
class PatchSplitModel(nn.Module):
    def __init__(self, model, image_size, split_num, stride_num=2, output_size=None,
                 patch_combine_method="region_voting", img_dim=2, process_at_once=32):
        super().__init__()
        method_list = ["region_voting", "eulidian_weight"]
        assert patch_combine_method in method_list, f"check combine_method in {method_list}"
        if output_size == None:
            output_size = image_size
        self.model = model
        self.process_at_once = process_at_once
        patch_size = image_size // split_num
        stride = patch_size // stride_num
        pad_size = stride
        output_patch_size = output_size // split_num
        output_stride = output_patch_size // stride_num
        output_pad_size = output_stride
        self.extract_patches = partial(extract_patch_tensor, patch_size=patch_size, stride=stride, pad_size=pad_size)
        if patch_combine_method == "region_voting":
            self.combine_region_voting_patches = partial(combine_region_voting_patches, image_size=output_size,
                                                         patch_size=output_patch_size, stride=output_stride, pad_size=output_pad_size, img_dim=img_dim)
        elif patch_combine_method == "eulidian_weight":
            self.combine_region_voting_patches = partial(combine_region_voting_patches_with_patch_weights, image_size=output_size,
                                                         patch_size=output_patch_size, stride=output_stride, pad_size=output_pad_size, img_dim=img_dim)
    def forward(self, x):
        batch_size = x.size(0)
        x_patches = self.extract_patches(x)
        if self.process_at_once is None:
            y_pred_patches = self.model(x_patches)
        else:
            y_pred_patches = process_patch_array(x_patches, self.model, self.process_at_once)
        y_pred_recon = self.combine_region_voting_patches(y_pred_patches, batch_size=batch_size)
        return y_pred_recon
    
    def compute_patch_output(self, x, is_full=False):
        if is_full:
            x_patches = self.extract_patches(x)
        else:
            x_patches = x
        y_pred_patches = self.model(x_patches)
        return y_pred_patches
    
    def compute_loss(self, x, y, loss_fn, is_y_full=False, use_model=True):
        if is_y_full:
            y_patches = self.extract_patches(y)
        else:
            y_patches = y
        if use_model:
            x_patches = self.extract_patches(x)
            y_pred_patches = self.model(x_patches)
        else:
            y_pred_patches = x
        return loss_fn(y_pred_patches, y_patches)
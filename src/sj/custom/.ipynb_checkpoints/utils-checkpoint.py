# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""utility functions."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

import os
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pydicom
import pandas as pd
import pickle5 as pickle
from scipy import ndimage
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager
from PIL import Image
# from skimage.metrics import structural_similarity as ssim

import random

np.random.seed(0)
random.seed(0)

DPI = 300
CLUSTER_SIZE = 600
# ----------------------------------------------------------------------------
# Image utils.


def normalize_hu(img, min_hu=-20, max_hu=180):
    """
    Normalize the img to represent HU value
    """
    def in_range(img, min=None, max=None):
        if min == None:
            return img * (img < max)
        if max == None:
            return img * (img > min)
        else:
            return img * (img >= min) * (img <= max)
    b, c, h, w = img.shape
    sample = img.clone()
    sample = torch.clamp(sample, -1, 1)

    sample_0 = 50 * (sample[:, 0] + 1) / 2 + 5
    sample_1 = 80 * (sample[:, 1] + 1) / 2
    sample_2 = 200 * (sample[:, 2] + 1) / 2 - 20

    sample = (in_range(sample_2, max=0) +
              (in_range(sample_1, 0, 5) + in_range(sample_2, 0, 5)) / 2 +
              (in_range(sample_0, 5, 55) + in_range(sample_1, 5, 55) + in_range(sample_2, 5, 55)) / 3 +
              (in_range(sample_1, 55, 80) + in_range(sample_2, 55, 80)) / 2 +
              in_range(sample_2, min=80))

    return sample


def windowing_brain(npy, channel=3, return_uint8=True):
    dcm = npy.copy()
    img_rows = 512
    img_cols = 512

    if channel == 1:
        npy = npy.squeeze()
        npy = cv2.resize(npy, (512, 512), interpolation=cv2.INTER_LINEAR)
        npy = npy + 40
        npy = np.clip(npy, 0, 160)
        npy = npy / 160

    elif channel == 3:
        dcm0 = dcm[0] - 5
        dcm0 = np.clip(dcm0, 0, 50)
        dcm0 = dcm0 / 50.

        dcm1 = dcm[0] + 0
        dcm1 = np.clip(dcm1, 0, 80)
        dcm1 = dcm1 / 80.

        dcm2 = dcm[0] + 20
        dcm2 = np.clip(dcm2, 0, 200)
        dcm2 = dcm2 / 200.

        npy = np.stack([dcm0, dcm1, dcm2], 2)

    if return_uint8:
        return np.uint8(npy * (2 ** 8 - 1))

    else:
        return npy


def windowing_thorax(img_png, npy, channel=3):
    dcm = npy.copy()
    img_rows = 512
    img_cols = 512

    if channel == 1:
        npy = npy.squeeze()
        npy = cv2.resize(npy, (512, 512), interpolation=cv2.INTER_LINEAR)
        npy = npy + 40  # change to lung/med setting
        npy = np.clip(npy, 0, 160)
        npy = npy / 160
        npy = 255 * npy
        npy = npy.astype(np.uint8)

    elif channel == 3:
        dcm1 = dcm[0] + 150
        dcm1 = np.clip(dcm1, 0, 400)
        dcm1 = dcm1 / 400.
        dcm1 *= (2 ** 8 - 1)
        dcm1 = dcm1.astype(np.uint8)

        dcm2 = dcm[0] - 250
        dcm2 = np.clip(dcm2, 0, 100)
        dcm2 = dcm2 / 100.
        dcm2 *= (2 ** 8 - 1)
        dcm2 = dcm2.astype(np.uint8)

        dcm3 = dcm[0] + 950
        dcm3 = np.clip(dcm3, 0, 1000)
        dcm3 = dcm3 / 1000.
        dcm3 *= (2 ** 8 - 1)
        dcm3 = dcm3.astype(np.uint8)

        npy = np.zeros([img_rows, img_cols, 3], dtype=int)
        npy[:, :, 0] = dcm2
        npy[:, :, 1] = dcm1
        npy[:, :, 2] = dcm3

    return npy


def write_png_image(img_png, npy):
    if not os.path.exists(img_png):
        return cv2.imwrite(img_png, npy)
    else:
        return False


def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
            np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data


# ----------------------------------------------------------------------------

# Image utils to save figures

def smooth_image(image, kernel_x=5, iters=1):
    iters -= 1
    kernel = np.ones((kernel_x, kernel_x), np.float32) / (kernel_x ** 2)

    if iters == 0:
        return cv2.filter2D(image, -1, kernel)
    else:
        return smooth_image(cv2.filter2D(image, -1, kernel), iters=iters)


def mean_filter(tensor, k=5, iters=1):
    iters -= 1
    b, c, h, w = tensor.shape
    filters = torch.ones(c, c, k, k).cuda() / (k * k)
    out = F.conv2d(tensor, filters, padding=(k - 1) // 2)
    if iters:
        return mean_filter(out, iters=iters)
    else:
        return out

# ----------------------------------------------------------------------------


def convert_to_numpy_array(image, drange=[0, 1], rgbtogray=False):
    assert image.ndim == 2 or image.ndim == 3
    if image.ndim == 3:
        if image.shape[0] == 1:
            image = image[0]  # grayscale CHW => HW
        else:
            image = image.transpose(1, 2, 0)  # CHW -> HWC

    image = adjust_dynamic_range(image, drange, [0, 255])
    image = np.rint(image).clip(0, 255).astype(np.uint8)

    if rgbtogray:
        return convert_rgb_to_gray(image)

    return image


def convert_to_pil_image(image, drange=[0, 1]):
    image = convert_to_numpy_array(image, drange)
    fmt = 'RGB' if image.ndim == 3 else 'L'
    return Image.fromarray(image, fmt)


def rgb2gray(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


def tensor_to_np(tensor):
    return (
        tensor.detach()
              .to('cpu')
              .numpy()
    )


def make_image(tensor):
    return (
        tensor.clone()
        .detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to('cpu')
        .numpy()
        .astype(np.uint8)
    )


def convert_rgb_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def apply_mirror_augment(minibatch):
    mask = np.random.rand(minibatch.shape[0]) < 0.5
    minibatch = np.array(minibatch)
    minibatch[mask] = minibatch[mask, :, :, ::-1]
    return minibatch

# ----------------------------------------------------------------------------


def create_summary_figure(real, fake, target, pred):
    real_gray = rgb2gray(real)
    fake_gray = rgb2gray(fake)

    figures = 4
    figure_size = 6
    # plt.rcParams["font.family"] = "Times New Roman"
    fontsize = 24
    fig = plt.figure(figsize=(figure_size * figures, figure_size))

    """First Figure : Real """
    fig.add_subplot(1, figures, 1)
    plt.title('Real', fontsize=fontsize)
    plt.axis('off')
    plt.imshow(real_gray, cmap="gray")  # transpose(real))

    """Second Figure : Input + Target """
    fig.add_subplot(1, figures, 2)
    plt.axis('off')
    plt.title("Target", fontsize=fontsize)
    plt.imshow(real_gray, cmap="gray")

    # target = target.astype(np.int) * 255
    plt.imshow(colorize_mask(target), alpha=0.75)

    """Third Figure : Fake """
    fig.add_subplot(1, figures, 3)
    plt.title('Fake', fontsize=fontsize)
    plt.axis('off')
    plt.imshow(fake_gray, cmap="gray")

    """4th : Prediction for lesion"""
    fig.add_subplot(1, figures, 4)
    plt.axis('off')
    plt.title("Prediction", fontsize=fontsize)
    plt.imshow(real_gray, cmap="gray")

    plt.imshow(pred, alpha=0.75)
    fig.tight_layout()

    return fig


def save_summary_figure(real, fake, overlay, residue, residue_bet, residue_filtered, figure_save_dir, fName, save_format='png'):
    # real_gray = real
    # fake_gray = fake
    residual = np.abs(real - fake)
    fname = fName.replace('.dcm', '.png')

    mkdirs(os.path.join(figure_save_dir, 'input'))
    input_path = os.path.join(figure_save_dir, "input", fname)
    array2image(rgb2gray(real), input_path)

    mkdirs(os.path.join(figure_save_dir, 'reconstruction'))
    recon_path = os.path.join(figure_save_dir, "reconstruction", fname)
    array2image(rgb2gray(fake), recon_path)

    mkdirs(os.path.join(figure_save_dir, 'residual'))
    residue_path = os.path.join(figure_save_dir, "residual", fname)
    array2image(np.squeeze(residue, 2), residue_path, colorize=True)

    mkdirs(os.path.join(figure_save_dir, 'residual_bet'))
    residue_path = os.path.join(figure_save_dir, "residual_bet", fname)
    array2image(np.squeeze(residue_bet, 2), residue_path, colorize=True)

    mkdirs(os.path.join(figure_save_dir, 'residue_filtered'))
    residue_filtered_path = os.path.join(
        figure_save_dir, "residue_filtered", fname)
    array2image(np.squeeze(residue_filtered, 2),
                residue_filtered_path, colorize=True)

    overlay_path = os.path.join(figure_save_dir, "overlay", fname)
    mkdirs(os.path.join(figure_save_dir, 'overlay'))
    array2image(overlay, path=overlay_path,)

    input_overlaid_path = os.path.join(
        figure_save_dir, "input_overlaid", fname)
    mkdirs(os.path.join(figure_save_dir, 'input_overlaid'))

    overlay = cv2.imread(overlay_path, 0)
    _, img = cv2.threshold(
        overlay, 127, 255, cv2.THRESH_BINARY)  # ensure binary
    num_labels, labels = cv2.connectedComponents(img, connectivity=8)

    red_overlay = np.zeros(labels.shape)
    yellow_overlay = np.zeros(labels.shape)
    for label in range(1, np.max(labels) + 1):
        label_cluster_size = np.sum(labels == label)
        labels[labels == label] = 255 * (label_cluster_size > CLUSTER_SIZE)
        if (residual[:, :, 2] * (labels == label)).sum():
            red_overlay[labels == label] = 255 * \
                (label_cluster_size > CLUSTER_SIZE)
            yellow_overlay[labels == label] = 0
        else:
            red_overlay[labels == label] = 0
            yellow_overlay[labels == label] = 255 * \
                (label_cluster_size > CLUSTER_SIZE)

    labels = labels.astype(np.uint8)
    overlay = np.stack([labels, labels * 0, labels * 0], 2)

    array2image(overlay, path=overlay_path, colorize=False, transparent=True)
    transparent_blend(input_path, overlay_path, input_overlaid_path)


def minmax_normalization(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def standardize(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def artifact_mask(dcm, threshold=800):
    mask = dcm > threshold
    return mask


class FigureGenerator():
    def __init__(self):
        pass


def colorize_mask(mask, color="Red", masked=True):
    assert mask.shape == (512, 512), print(mask.shape)
#     mask = (255 * mask).astype(np.uint8)
    if color == "Yellow":
        mask = np.stack([mask * 1, mask * 1, mask * 0], 2)
    if color == "White":
        mask = np.stack([mask * 1, mask * 1, mask * 1], 2)
    if color == "Red":
        mask = np.stack([mask * 1, mask * 0, mask * 0], 2)
    # if color == "":
    #     mask = np.stack([mask * 1, mask * 1, mask * 0], 2)

    return mask
    # if masked: return np.ma.masked_where(mask == 0, mask)
    # else:      return mask


def transparent_blend(im1_path, im2_path, save_path, alpha=0.66):
    background = Image.open(im1_path).convert('RGB')
    foreground = Image.open(im2_path).convert('RGBA')
    foreground_trans = Image.new("RGBA", foreground.size)
    foreground_trans = Image.blend(foreground_trans, foreground, alpha)

    background.paste(foreground_trans, (0, 0), foreground_trans)
    background.save(save_path, dpi=(DPI, DPI))

    # # for np array
    # img_pil     = array2image(img)
    # array2image(mask, path = path, colorize = True)
    # # mask_pil_tr = Image.new("RGBA", mask_pil.size)

    # mask_pil_tr = array2image(mask, colorize = True)

    # # mask_pil_tr = Image.blend(mask_pil_tr, mask_pil, 1.0)

    # img_overlaid = Image.blend(img_pil, mask_pil_tr, 0.5)
    # # img_pil.paste(mask_pil_tr, (0,0), mask_pil_tr)
    # if path:
    #     img_overlaid.save(path)


def transparent_mask(pil):
    rgba = pil.convert("RGBA")
    rgba_data = rgba.getdata()

    newData = []
    for item in rgba_data:
        # finding black colour by its RGB value
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            # storing a transparent value when we find a black colour
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)  # other colours remain unchanged

    rgba.putdata(newData)
    return rgba


def get_patient_info(dcm, force=True):
    dcm_data = pydicom.dcmread(dcm, force=force)
    patient_id = dcm_data.PatientID if hasattr(dcm_data, "PatientID") else ""
    age = dcm_data.PatientAge if hasattr(dcm_data, "PatientAge") else 0
    age = int(age[:-1]) if age else 0
    sex = dcm_data.PatientSex if hasattr(dcm_data, "PatientSex") else ""
    return patient_id, age, sex


def get_dicom(dcm, force=True):
    dcm_data = pydicom.dcmread(dcm, force=force)

    img = dcm_data.pixel_array.astype(np.float)
    img = crop_center(np.expand_dims(img, 0))
    if windowing:
        img = windowing_brain(img - 1024)
    else:
        img = img / 2 ** 12  # normalize from [0, 4096] to [0, 1]

    age = dcm_data.PatientAge if hasattr(dcm_data, "PatientAge") else 0
    age = int(age[:-1]) if age else 0
    sex = dcm_data.PatientSex if hasattr(dcm_data, "PatientSex") else ""
    return img, age, sex


def diff_score(img_reals, img_fakes, bet_masks):
    b, c, h, w = img_reals.shape
    diff = torch.abs(img_reals - img_fakes) * \
        bet_masks.unsqueeze(1)  # simple residual difference
    """
    _,diff = ssim(real_tmp, fake_tmp, full = True)
    
    about SSIM: 
        -   https://bskyvision.com/396
        -   https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
        -   Don't know why but it is worse than residual difference..
            -   SSIM is for "structural" similarity, SSIM considers white matter and gray matter as "noise"
    """
    return diff.reshape(b, -1).mean(1)

# def gaussian_filter(kernel_size = 15, sigma = 3):

#     conv = torch.nn.Conv2d(..., bias = False)
#     with torch.no_grad():
#         conv_weight = gaussian_weights


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=True):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(
            3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


class MedianPool(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    modified from https://gist.github.com/keunwoochoi/dcbaf3eaa72ca22ea4866bd5e458e32c
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self,
                 kernel_size: tuple = (1, 17, 17),
                 stride: tuple = (1, 1, 1),
                 padding: int = 0,
                 same: bool = True):
        super(MedianPool, self).__init__()

        self.k = kernel_size
        self.stride = stride
        self.padding = padding  # convert to l, r, t, b
        self.same = same

    @torch.no_grad()
    def _padding(self, x):
        if self.same:
            padding = []
            for dimension, k, stride in zip(x.size()[1:], self.k, self.stride):
                if dimension % stride == 0:
                    pad = max(k - stride, 0)
                else:
                    pad = max(k - (dimension % stride), 0)

                padding.append(pad // 2)
                padding.append(pad - (pad // 2))
            padding.reverse()
            padding = tuple(padding)
        else:
            padding = _quadruple(self.padding)
        return padding

    @torch.no_grad()
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level

        # 1. padding
        x = F.pad(x, self._padding(x), mode="constant", value=0.0)

        # 2. unfold
        for i, (k, stride) in enumerate(zip(self.k, self.stride)):
            x = x.unfold(i + 1, k, stride)

        # 3. median operation
        # Note that view() requires tensor to be contiguously stored for fast reshape operation in memory.
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]

        return x


def voting(diff_maps, voting_maps, thres):
    votes = (diff_maps > thres).float()

    votes = torch.zeros_like(diff_maps)
    for level in [1, 2, 3]:
        votes += (diff_maps > level * thres).float()

    adder = torch.zeros_like(votes)
    for fNum in range(votes.size(0)):
        if fNum != 0 and (fNum + 1) != votes.size(0):
            adder[fNum] += ((votes[fNum - 1] * votes[fNum])
                            + (votes[fNum + 1] * votes[fNum])
                            + (votes[fNum - 1] * votes[fNum + 1]))
    votes += adder.bool().float()

    return voting_maps + votes


def mask_cluster(masks, threshold):
    masks_np = masks.clone().detach().cpu().numpy()
    for i in range(len(masks_np)):
        mask = masks_np[i]
        _, mask = cv2.threshold(mask, threshold, 255,
                                cv2.THRESH_BINARY)  # ensure binary
        num_labels, labels = cv2.connectedComponents(
            mask.astype(np.uint8), connectivity=8)
        for label in range(1, np.max(labels) + 1):
            label_cluster_size = np.sum(labels == label)
            labels[labels == label] = (label_cluster_size > CLUSTER_SIZE)
        masks_np[i] = labels
        with torch.no_grad():
            masks[i] = torch.from_numpy(masks_np[i])
    return masks.bool()

# def voting(diff_maps, voting_maps, step, thres = 5):
#    votes = (diff_maps > thres).bool()
#
#    for fNum in range(votes.size(0)):
#        if fNum != 0 and (fNum+1) != votes.size(0):
#            votes[fNum] += votes[fNum-1] * votes[fNum+1]
#
#    voting_maps *= votes
#    return voting_maps

# def voting(diff_maps, preds, thres = 5, vote = True):
#     if vote:
#         preds *= (diff_maps[:,0] > thres).type(torch.bool)
#     else:
#         preds  = (diff_maps[:,0] > thres).type(torch.bool)

#     for _ in range (32):
#         for fNum in range(preds.size(0)):
#             if fNum != 0 and (fNum+1) != preds.size(0):
#                 preds[fNum] += preds[fNum-1] * preds[fNum+1]

#     return preds


# def to256(img):
#     batch, channel, height, width = img.shape
#     img_256 = img.clone()

#     if height > 256:
#         factor = height // 256

#         img_256 = img_256.reshape(
#             batch, channel, height // factor, factor, width // factor, factor
#         ).mean([3, 5])

#     return img_256

def to256(img):
    batch, channel, height, width = img.shape

    if height > 256:
        factor = height // 256

        return img.view(batch,
                        channel,
                        height // factor,
                        factor,
                        width // factor,
                        factor).mean([3, 5])

    return img


def shrink(img, size):
    batch, channel, height, width = img.shape
    img_shrink = img.clone()

    if height > size:
        factor = height // size

        img_shrink = img_shrink.reshape(
            batch, channel, height // factor, factor, width // factor, factor
        )
        img_shrink = img_shrink.mean([3, 5])

    return img_shrink


def save_obj(obj, PATH):
    with open(PATH, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(PATH, printf=False):
    import pickle5 as pickle

    if os.path.exists(PATH):
        if printf:
            print(f"INFO.loaded {stat_pkl}")
        with open(PATH, 'rb') as f:
            return pickle.load(f)

    else:
        print(f"INFO.{PATH} does not exist!")
        return None


def mkdir(*args):
    if not os.path.exists(*args):
        os.mkdir(*args)


def mkdirs(path):
    assert path[0] != '/'
    for i in range(len(path.split('/'))):
        mkdir(os.path.join(*path.split('/')[:i + 1]))


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return torch.utils.data.RandomSampler(dataset)

    else:
        return torch.utils.data.SequentialSampler(dataset)


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          save_path=""):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    augments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()

    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))
    if save_path:
        plt.savefig(save_path, dpi=DPI, transparent=True)
    plt.show()
    plt.close()


def array2image(arr, path="", colorize=False, transparent=False):
    if colorize:
        arr = colorize_mask(arr)

    im = Image.fromarray(arr)

    if transparent:
        im = transparent_mask(im)

    if path:
        im.save(path, dpi=(DPI, DPI))

    return im


def show_tensor_images(image_tensor):
    n, c, h, w = image_tensor.shape
    image_tensor = image_tensor + 1 / 2
    image = image_tensor.detach().cpu()
    image_grid = make_grid(image, nrow=int(n ** 0.5))
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    return plt.show()


def make_excel(data, filename):
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)


def make_mutiple_excel(dataset, filename, sheet_names=[]):
    ##############################################################################
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')

    # Write each dataframe to a different worksheet.
    if sheet_names:
        for data, sheet_name in zip(dataset, sheet_names):
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=sheet_name)
    else:
        for i, data in enumerate(dataset):
            df = pd.ExcelWriter(data)
            df.to_excel(writer, sheet_name=f"sheet{str(i)}")

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


def dcm2jpg():
    jpg_dir = os.path.join("Brain_CT_Data/test/internal_validation/jpg")
    mkdir(jpg_dir)
    for case in ['abnormal', 'normal', 'benign']:
        brainct_dirs = glob(os.path.join("Brain_CT_Data/test", case, "*",))
        case_jpg_dir = os.path.join(jpg_dir, case)
        mkdir(case_jpg_dir)
        print(f"[INFO]change {case} dcm to jpg files")

        for brainct_dir in tqdm(brainct_dirs):

            brain_imgs = []
            dcms = sorted(glob(os.path.join(brainct_dir, '2', "*.dcm")))
            for dcm in dcms:
                my_dcm = pydicom.dcmread(dcm)
                "append image"
                arr = my_dcm.pixel_array.astype(np.int32) - 1024
                brain_img = windowing_brain(arr)
                brain_imgs.append(brain_img)

            plt.rcParams['figure.facecolor'] = 'black'
            fig = plt.figure(figsize=(32, 32))
            columns_ = int(len(brain_imgs)**0.5)
            rows_ = math.ceil(len(brain_imgs) / columns_)

            for i, brain_img in enumerate(brain_imgs):
                fig.add_subplot(rows_, columns_, i + 1)
                plt.imshow(brain_img)

            plt.axis('off')
            plt.axis("tight")

            patient_id = brainct_dir.split('/')[-1]
            plt.savefig(os.path.join(
                case_jpg_dir, f"{patient_id}.jpg"), dpi=DPI, transparent=True)
            plt.close()

            if case == "abnormal":
                mask_hdrs = glob(os.path.join(
                    brainct_dir, 'Mask_DataSet', "*.hdr"))
                targets = np.zeros([len(brain_imgs), 512, 512])
                for mask_hdr in mask_hdrs:
                    targets += sitk.GetArrayFromImage(sitk.ReadImage(mask_hdr))

                plt.rcParams['figure.facecolor'] = 'black'
                fig = plt.figure(figsize=(32, 32))
                columns_ = int(len(brain_imgs)**0.5)
                rows_ = math.ceil(len(brain_imgs) / columns_)

                for i, (brain_img, target) in enumerate(zip(brain_imgs, targets)):
                    fig.add_subplot(rows_, columns_, i + 1)
                    plt.imshow(brain_img)
                    plt.imshow(np.flip(target, 0).astype(bool),
                               alpha=0.7, cmap=plt.cm.inferno)
                plt.axis('off')
                plt.axis("tight")

                plt.savefig(os.path.join(
                    case_jpg_dir, f"{patient_id}_target.jpg"), dpi=DPI, transparent=True)
                plt.close()


def crop_center(img, cropx=512, cropy=512):
    b, y, x = img.shape
    if y < cropy or x < cropx:
        return np.zeros([b, cropx, cropy])
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2

    return img[:, starty:starty + cropy, startx:startx + cropx]


def dcm2img(dcm, windowing=True, return_uint8=True):
    dcm_data = pydicom.dcmread(dcm,
                               force=True)
    # InvalidDicomError: File is missing DICOM File Meta Information header or the 'DICM' prefix is missing from the header. Use force=True to force reading.

    if hasattr(dcm_data, "pixel_array"):
        img = dcm_data.pixel_array.astype(np.float)
        img = crop_center(np.expand_dims(img, 0))
    else:
        if windowing:
            img = np.zeros([3, 512, 512])
        else:
            img = np.zeros([1, 512, 512])
    if windowing:
        return windowing_brain(img - 1024, return_uint8=return_uint8)
    else:
        return (img - 1024) / 2 ** 12


def dcm2np(dcm, windowing=True, return_uint8=True):
    img = sitk.GetArrayFromImage(sitk.ReadImage(dcm))
    img = crop_center(img)

    if windowing:
        return windowing_brain(img, return_uint8=return_uint8)
    else:
        return img / 2 ** 12


def png2arr(png, rgb=True):
    np_frame = np.array(Image.open(png))
    if rgb:
        return np.uint8(np_frame)
    else:
        return rgb2gray(np.uint8(np_frame))

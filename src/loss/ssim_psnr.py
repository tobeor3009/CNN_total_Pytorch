import torch
import torch.nn.functional as F
from functools import partial
from .util import UnexpectedBehaviorException

def get_filter_fn(win_size, img_dim):
    if img_dim == 2:
        filter_fn = F.avg_pool2d
    elif img_dim == 3:
        filter_fn = F.avg_pool3d
    return partial(filter_fn, kernel_size=win_size, stride=1, padding=win_size // 2)

# expected input shape is [B, C, H, W or B, C, Z, H, W]
def structural_similarity_torch(
    im1,
    im2,
    win_size=None,
    data_range=1.0,
    full=False,
    filter_fn=None
):
    batch_size, image_channel, *img_size_list = im1.shape
    img_dim = len(img_size_list)
    target_mean_dim_tuple = tuple(range(2, 2 + img_dim))

    # Ensure the input tensors are float32
    im1 = im1.float()
    im2 = im2.float()
    if data_range is None:
        raise UnexpectedBehaviorException("you need to set data_range, example: 1.0, meaning data interval")
    if win_size is None:
        win_size = 7

    # Default values for constants
    K1 = 0.01
    K2 = 0.03
    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    
    # Gaussian weights or uniform filter
    if filter_fn is None:
        filter_fn = get_filter_fn(win_size, img_dim)

    NP = win_size ** img_dim
    conv_norm = NP / (NP - 1)
    # Compute means
    ux = filter_fn(im1)
    uy = filter_fn(im2)

    # Compute variances and covariances
    uxx = filter_fn(im1 * im1)
    uyy = filter_fn(im2 * im2)
    uxy = filter_fn(im1 * im2)
    
    vx = conv_norm * (uxx - ux * ux)
    vy = conv_norm * (uyy - uy * uy)
    vxy = conv_norm * (uxy - ux * uy)

    A1, A2, B1, B2 = (
        2 * ux * uy + C1,
        2 * vxy + C2,
        ux**2 + uy**2 + C1,
        vx + vy + C2,
    )
    D = B1 * B2
    S = (A1 * A2) / D

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2
    all_slice = slice(None, None)
    crop_slice = slice(pad, -pad)
    image_crop_slice_tuple = tuple(crop_slice for _  in target_mean_dim_tuple)
    index_tuple = (all_slice, all_slice, *image_crop_slice_tuple)
    cropped_result = S[index_tuple]
    
    mssim = cropped_result.mean(target_mean_dim_tuple).mean()
    
    if full:
        return mssim, S
    else:
        return mssim

class SSIM(torch.nn.Module):
    def __init__(self, win_size=None, data_range=1.0, full=False):
        super(SSIM, self).__init__()
        self.win_size = win_size
        self.data_range = data_range
        self.full = full

    def forward(self, img1, img2):
        return structural_similarity_torch(img1, img2, self.win_size, 
                                           self.data_range, self.full)


def get_ssim(img1, img2, win_size=None, data_range=1.0, full=False):
    return structural_similarity_torch(img1, img2, win_size,
                                       data_range, full)

def peak_signal_noise_ratio_pytorch(image_true, image_test, data_range=None):
    batch_size, image_channel, *img_size_list = image_true.shape
    img_dim = len(img_size_list)
    target_mean_dim_tuple = tuple(range(2, 2 + img_dim))

    image_true = image_true.float()
    image_test = image_test.float()
    if data_range is None:
        raise UnexpectedBehaviorException("you need to set data_range, example: 1.0, meaning data interval")
    
    err = (image_true - image_test) ** 2
    err = err.mean(target_mean_dim_tuple)
    psnr_score = 10 * torch.log10((data_range ** 2) / err)
    return psnr_score.mean()

class PNSR(torch.nn.Module):
    def __init__(self, data_range=1.0):
        super(SSIM, self).__init__()
        self.data_range = data_range

    def forward(self, img1, img2):
        return peak_signal_noise_ratio_pytorch(img1, img2, self.data_range)


def get_psnr(img1, img2, data_range=1.0):
    return peak_signal_noise_ratio_pytorch(img1, img2, data_range)
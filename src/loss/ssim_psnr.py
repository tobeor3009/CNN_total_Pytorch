from math import exp
import torch
import torch.nn.functional as F
from functools import partial
from torch.autograd import Variable

from .util import UnexpectedBehaviorException
from ..util.fold_unfold import extract_patch_tensor

DEFAULT_DATA_RANGE = 1.0
DEFAULT_REDUCTION = "mean"
SSIM_VERSION_LIST = ["pytorch", "skimages"]
# code from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel, img_dim):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _nD_window = _1D_window.mm(_1D_window.t()).float()[tuple(None for _ in range(img_dim))]
    window = Variable(_nD_window.expand(channel, 1, *(window_size for _ in range(img_dim))).contiguous())
    return window

def get_conv_fn(window_size, channel, img_dim, device):
    window = create_window(window_size, channel, img_dim).to(device)
    if img_dim == 2:
        conv_fn = F.conv2d
    else:
        conv_fn = F.conv3d
    return partial(conv_fn, weight=window, padding=window_size // 2, groups=channel)

def _ssim(img1, img2, window, window_size, channel, conv_fn, img_dim, data_range, reduction):
    
    target_mean_dim_tuple = tuple(range(2, 2 + img_dim))
    mu1 = conv_fn(img1, window, padding = window_size//2, groups = channel)
    mu2 = conv_fn(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = conv_fn(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = conv_fn(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = conv_fn(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = (0.01 * data_range) **2
    C2 = (0.03 * data_range) **2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    ssim_map = ssim_map.mean(target_mean_dim_tuple)
    if reduction == "mean":
        ssim_map = ssim_map.mean()
    elif reduction == "none":
        ssim_map = ssim_map.mean(1)
    return ssim_map

def get_ssim_pytorch(img1, img2, window_size = 7, data_range=DEFAULT_DATA_RANGE, reduction=DEFAULT_REDUCTION):
    (_, channel, *img_size_list) = img1.size()
    img_dim = len(img_size_list)
    if img_dim == 2:
        conv_fn = F.conv2d
    else:
        conv_fn = F.conv3d
    window = create_window(window_size, channel, img_dim)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, conv_fn, img_dim, data_range, reduction)

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
    data_range=DEFAULT_DATA_RANGE,
    full=False,
    filter_fn="avg",
    reduction=DEFAULT_REDUCTION
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
    if filter_fn == "avg":
        filter_fn = get_filter_fn(win_size, img_dim)
    if filter_fn == "conv":
        filter_fn = get_conv_fn(win_size, image_channel, img_dim, im1.device)
        
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

    mssim = cropped_result.mean(target_mean_dim_tuple)
    if reduction == "mean":
        mssim = mssim.mean()
    elif reduction == "none":
        mssim = mssim.mean(1)
    if full:
        return mssim, S
    else:
        return mssim

class SSIM(torch.nn.Module):
    def __init__(self, win_size=None, data_range=DEFAULT_DATA_RANGE, full=False, filter_fn="avg", reduction=DEFAULT_REDUCTION):
        super(SSIM, self).__init__()
        self.win_size = win_size
        self.data_range = data_range
        self.full = full
        self.filter_fn = filter_fn
        self.reduction = reduction

    def forward(self, img1, img2):
        return structural_similarity_torch(img1, img2, win_size=self.win_size,
                                           data_range=self.data_range, full=self.full, 
                                           filter_fn=self.filter_fn, reduction=self.reduction)


def get_ssim(img1, img2, win_size=None, data_range=1.0, full=False, filter_fn="avg", reduction="mean", version="pytorch"):
    
    assert version in SSIM_VERSION_LIST, f"support ssim version is {SSIM_VERSION_LIST}"
    if version == "pytorch":
        return get_ssim_pytorch(img1, img2, window_size=win_size,
                                data_range=data_range, reduction=reduction)
    elif version == "skimages":
        return structural_similarity_torch(img1, img2, win_size=win_size,
                                          data_range=data_range, full=full, filter_fn=filter_fn, reduction=reduction)

def peak_signal_noise_ratio_pytorch(image_true, image_test, data_range=DEFAULT_DATA_RANGE, reduction=DEFAULT_REDUCTION):
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
    if reduction == "mean":
        psnr_score = psnr_score.mean()
    elif reduction == "none":
        psnr_score = psnr_score.mean(1)
    return psnr_score

class PNSR(torch.nn.Module):
    def __init__(self, data_range=DEFAULT_DATA_RANGE, reduction=DEFAULT_REDUCTION):
        super(SSIM, self).__init__()
        self.data_range = data_range
        self.reduction = reduction
    def forward(self, img1, img2):
        return peak_signal_noise_ratio_pytorch(img1, img2, data_range=self.data_range, reduction=self.reduction)


def get_psnr(img1, img2, data_range=DEFAULT_DATA_RANGE, reduction=DEFAULT_REDUCTION):
    return peak_signal_noise_ratio_pytorch(img1, img2, data_range=data_range, reduction=reduction)

def get_max_ssim_loss_fn(y_pred, y_true, data_range=DEFAULT_DATA_RANGE, image_to_patch_ratio=4):
    batch_size, _, *image_size_list = y_pred.shape
    
    image_size = int(image_size_list[-1])
    patch_size = image_size // image_to_patch_ratio
    stride = patch_size // 4
    num_patch = int((image_size - patch_size) / stride) + 1
    
    # y_pred_patch.shape = [B * (num_patch ** 2), C, H, W]
    y_pred_patch = extract_patch_tensor(y_pred, patch_size, stride, pad_size=0)
    y_true_patch = extract_patch_tensor(y_true, patch_size, stride, pad_size=0)

    ssim_score_patch = get_ssim(y_pred_patch, y_true_patch, data_range=data_range, reduction="none", filter_fn="conv", version="skimages")
    ssim_score_patch = ssim_score_patch.view(batch_size, num_patch ** 2)
    ssim_min_k_score_patch = torch.topk(ssim_score_patch, num_patch, dim=1, largest=False).values
    # ssim_min_k_score_patch.shape = [B, num_patch]
    ssim_max_k_loss_patch = -ssim_min_k_score_patch
    ssim_max_loss = ssim_max_k_loss_patch.mean()
    return ssim_max_loss

def get_min_psnr_fn(y_pred, y_true, data_range=DEFAULT_DATA_RANGE, image_to_patch_ratio=4):
    batch_size, _, *image_size_list = y_pred.shape
    
    image_size = int(image_size_list[-1])
    patch_size = image_size // image_to_patch_ratio
    stride = patch_size // 4
    num_patch = int((image_size - patch_size) / stride) + 1
    
    # y_pred_patch.shape = [B * (num_patch ** 2), C, H, W]
    y_pred_patch = extract_patch_tensor(y_pred, patch_size, stride, pad_size=0)
    y_true_patch = extract_patch_tensor(y_true, patch_size, stride, pad_size=0)

    psnr_score_patch = get_psnr(y_pred_patch, y_true_patch, data_range=data_range, reduction="none")
    psnr_score_patch = psnr_score_patch.view(batch_size, num_patch ** 2)
    psnr_min_k_score_patch = torch.topk(psnr_score_patch, num_patch, dim=1, largest=False).values
    # ssim_min_k_score_patch.shape = [B, num_patch]
    psnr_min_k_score = psnr_min_k_score_patch.mean()
    return psnr_min_k_score

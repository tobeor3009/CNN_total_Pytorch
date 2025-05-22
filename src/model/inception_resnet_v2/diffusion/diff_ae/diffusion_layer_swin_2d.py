
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from collections import namedtuple
from copy import deepcopy
from torch.utils.checkpoint import checkpoint
from timm.models.layers import DropPath, to_2tuple
import numpy as np
from .nn import zero_module
from .diffusion_layer import GroupNorm32, QKVAttention, QKVAttentionLegacy, ConvBlockND
from .diffusion_layer import PixelShuffle1D, PixelShuffle3D
from .diffusion_layer import get_conv_nd_fn, conv_nd, conv_transpose_kwarg_dict
from src.model.inception_resnet_v2.common_module.layers import get_act, get_norm, DEFAULT_ACT, INPLACE
from src.model.swin_transformer.model_2d.swin_layers import check_hasattr_and_init
from src.model.swin_transformer.model_2d.swin_layers import WindowAttention as WindowAttention2D
from src.model.swin_transformer.model_3d.swin_layers import WindowAttention as WindowAttention3D
from src.model.swin_transformer.model_2d.swin_layers import DROPOUT_INPLACE, Mlp
from .diffusion_layer import get_scale_shift_list, apply_embedding
from einops import rearrange, repeat
from functools import partial
from functools import wraps
from packaging import version
import itertools
from src.util.common import _ntuple
from src.model.inception_resnet_v2.diffusion.diff_ae.flash_attn import FlashMultiheadAttention

DEFAULT_ACT = get_act("silu")

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(target_list, t, x_shape):
    batch_size, *_ = t.shape
    out = target_list[t - 1]
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

def closed_form_sequence(n, img_dim):
    return math.floor(n / 2) + 1 + img_dim * (n % 2)

def window_partition(x, window_size_tuple):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, *img_size_list, C = x.shape
    img_dim = len(img_size_list)
    permute_tuple = tuple(closed_form_sequence(n, img_dim) for n in range(0, img_dim * 2))
    view_tuple = tuple(item for img_size, window_size in zip(img_size_list, window_size_tuple)
                      for item in (img_size // window_size, window_size))
    x = x.view(B, *view_tuple, C)
    windows = x.permute(0, *permute_tuple, -1).contiguous()
    windows = windows.view(-1, *window_size_tuple, C)
    return windows

def window_reverse(windows, window_size_tuple, img_resolution):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    img_dim = len(img_resolution)
    window_size_tuple = np.array(window_size_tuple)
    permute_tuple = tuple(closed_form_sequence(n, img_dim) for n in range(0, img_dim * 2))

    B = int(windows.shape[0] / (np.prod(img_resolution) / np.prod(window_size_tuple)))
    x = windows.view(B, *(img_resolution // window_size_tuple), *window_size_tuple, -1)
    x = x.permute(0, *permute_tuple, -1).contiguous().view(B, *img_resolution, -1)
    return x

def get_scale_shift_list(emb_block_list, emb_type_list, emb_args):
    scale_shift_list = []
    for emb_block, emb_type, emb in zip(emb_block_list, emb_type_list, emb_args):
        emb = emb_block(emb)
        emb = emb.unsqueeze(1)
        if emb_type == "cond":
            scale, shift = emb, 0
        else:
            scale, shift = emb.chunk(2, dim=-1)
        scale_shift_list.append([scale, shift])
    return scale_shift_list

def process_checkpoint_block(use_checkpoint, block, x, *emb_args):
    if use_checkpoint:
        output = checkpoint(block, x, *emb_args, use_reentrant=False)
    else:
        output = block(x, *emb_args)
    return output

class RMSNorm(nn.Module):
    def __init__(self, dim, img_dim=0, eps=1e-8):
        super().__init__()
        # img_dim = 0: sequence [B, N, D], img_dim = 1: 1d, img_dim = 2: 2d, img_dim = 3: 3d
        if img_dim == 0:
            param_shape = (1, 1, dim)
            self.normalize_dim = 2
        elif img_dim == 1:
            param_shape = (1, dim, 1)
            self.normalize_dim = 1
        elif img_dim == 2:
            param_shape = (1, dim, 1, 1)
            self.normalize_dim = 1
        elif img_dim == 3:
            param_shape = (1, dim, 1, 1, 1)
            self.normalize_dim = 1
        self.weight = nn.Parameter(torch.ones(*param_shape))
        self.eps = eps
        self.scale = dim ** 0.5

    def forward(self, x):
        rms = x.pow(2).mean(dim=self.normalize_dim, keepdim=True).add(self.eps).sqrt()
        x_normed = x / rms
        return x_normed * self.weight * self.scale

def get_rms_norm_nd(img_dim):
    norm_layer = partial(RMSNorm, img_dim=img_dim)
    return norm_layer

class MeanBN2BC(nn.Module):
    def __init__(self, target_dim=1):
        super(MeanBN2BC, self).__init__()
        self.target_dim = target_dim
    def forward(self, x):
        """
        x: Tensor of shape (B, N, C)
        Returns:
            Tensor of shape (B, C), where N-dimension is averaged out.
        """
        return x.mean(dim=self.target_dim)
    
class SelfAttention(nn.MultiheadAttention):
    def forward(self, x):
        output = super().forward(query=x, key=x, value=x, need_weights=False)
        return output[0]
class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered
    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)
    
class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x

class GroupNorm32(nn.GroupNorm):
    def __init__(self, num_channels: int, *args, **kwargs) -> None:
        num_groups = min(32, num_channels)
        super().__init__(num_groups, num_channels, *args, **kwargs)

def get_norm_layer(dim, norm_layer):

    if callable(norm_layer):
        norm_layer_instance = norm_layer(dim)
    else:
        norm_layer_str = norm_layer
        if norm_layer_str == "rms":
            norm_layer_instance = RMSNorm(dim)
        elif norm_layer_str == "group":
            norm_layer_instance = GroupNorm32(dim)
        elif norm_layer_str is None:
            norm_layer_instance = nn.Identity()
        else:
            raise NotImplementedError()
    return norm_layer_instance

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        norm_layer="group"
    ):  
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = get_norm_layer(channels, norm_layer)
        self.attention = FlashMultiheadAttention(dim=channels, num_heads=self.num_heads, causal=False)
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x, *args):
        if self.use_checkpoint:
            return checkpoint(self._forward_impl, x, *args,
                              use_reentrant=False)
        else:
            return self._forward_impl(x, *args)

    def _forward_impl(self, x, *args):
        b, n, c = x.shape
        x = self.norm(x)
        h = self.attention(x)
        h = self.proj_out(h)
        return (x + h)
    
def get_emb_block_list(act_layer, emb_dim_list, emb_type_list, dim):
    assert len(emb_dim_list) == len(emb_type_list)
    emb_block_list = []
    for emb_dim, emb_type in zip(emb_dim_list, emb_type_list):
        if emb_type == "seq":
            out_dim = dim * 2
        elif emb_type == "cond":
            out_dim = dim
        else:
            raise NotImplementedError()
        emb_block = nn.Sequential(
                                    act_layer,
                                    nn.Linear(emb_dim, out_dim)
                                )
        emb_block.append(emb_block)
    return nn.ModuleList(emb_block_list)

class PixelShuffleLinear(nn.Module):
    def __init__(self, upscale_factor):
        '''
        :param scale: upsample scale
        '''
        super().__init__()

        if isinstance(upscale_factor, int):
            upscale_factor = (upscale_factor, upscale_factor)
        self.scale_num = np.prod(upscale_factor)

    def forward(self, input_tensor):
        batch_size, component, channels = input_tensor.size()
        nOut = channels // self.scale_num

        out_component = component * self.scale_num

        input_view = input_tensor.view(batch_size, component, self.scale_num, nOut)

        output = input_view.permute(0, 2, 1, 3)
        output = output.contiguous()

        return output.view(batch_size, out_component, nOut)

class WindowContextAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))),
                                        requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))
        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1),
                                         self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1),
                                         self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.meshgrid([relative_coords_h,
                                                relative_coords_w], indexing='ij')
        relative_coords_table = torch.stack(relative_coords_table, dim=0)
        relative_coords_table = relative_coords_table.permute(
            1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :,
                                  0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :,
                                  1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h,
                                             coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - \
            1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)

        self.q_norm = RMSNorm(dim)
        self.k_norm = RMSNorm(dim)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop, inplace=DROPOUT_INPLACE)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=DROPOUT_INPLACE)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, context, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            kv_bias = torch.cat((torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        q = F.linear(input=x, weight=self.q.weight, bias=self.q_bias)
        kv = F.linear(input=context, weight=self.kv.weight, bias=kv_bias)
        
        # [B, N, H, C] => [B, H, N, C]
        # [B, N, 2, H, C] => [2, B, H, N, C]
        q = q.reshape(B_, N. self.num_heads, -1).permute(0, 2, 1 ,3).contiguous()
        kv = kv.reshape(B_, N, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        # make torchscript happy (cannot use tensor as tuple)
        k, v = kv[0], kv[1]

        # cosine attention
        attn = (self.q_norm(q) @ self.k_norm(k).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale,
                                  max=np.log(1. / 0.01)).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(
            self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class WindowAttention1D(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qkv_drop=0., attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0], cbp_dim=512):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))),
                                        requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(1, cbp_dim, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(cbp_dim, num_heads, bias=False))
        # get relative_coords_table
        relative_coords_table = torch.arange(-(self.window_size[0] - 1),
                                         self.window_size[0], dtype=torch.float32)
        relative_coords_table = torch.stack(relative_coords_table, dim=0)
        relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[..., 0] /= (pretrained_window_size[0] - 1)
        else:
            relative_coords_table[..., 0] /= (self.window_size[0] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords = torch.arange(self.window_size[0])
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (coords_flatten[:, :, None] -
                           coords_flatten[:, None, :])  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.qkv_drop = nn.Dropout(qkv_drop, inplace=DROPOUT_INPLACE)
        self.attn_drop = nn.Dropout(attn_drop, inplace=DROPOUT_INPLACE)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=DROPOUT_INPLACE)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(
                self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = self.qkv_drop(qkv)
        qkv = qkv.reshape(B_, N, 3,
                          self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # cosine attention
        attn = (F.normalize(q, dim=-1) @
                F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale,
                                  max=np.log(1. / 0.01)).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(
            self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0], -1)  # Wd, nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wd, Wd
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., qkv_drop=0., attn_drop=0., drop_path=0.,
                 norm_layer="rms", act_layer=DEFAULT_ACT, pretrained_window_size=0, img_dim=2):
        super().__init__()
        self.dim = dim
        self.input_resolution = np.array(input_resolution)
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.window_size_tuple = _ntuple(img_dim)(window_size)
        self.shift_size_tuple = _ntuple(img_dim)(shift_size)
        self.shift_dim_tuple = tuple(range(1, img_dim + 1))
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.window_size_tuple = _ntuple(img_dim)(min(self.input_resolution))
            self.shift_size_tuple = _ntuple(img_dim)(0)
        assert 0 <= min(self.shift_size_tuple) < min(self.window_size_tuple), "shift_size must in 0-window_size"

        self.norm1 = get_norm_layer(dim, norm_layer)
        if img_dim == 1:
            self.attn = WindowAttention1D(dim, window_size=self.window_size_tuple,
                                        num_heads=num_heads, qkv_bias=qkv_bias,
                                        qkv_drop=qkv_drop, attn_drop=attn_drop, proj_drop=drop,
                                        pretrained_window_size=_ntuple(img_dim)(pretrained_window_size),
                                        cbp_dim=np.clip(np.prod(input_resolution) // 16, 256, 1024))
        if img_dim == 2:
            self.attn = WindowAttention2D(dim, window_size=self.window_size_tuple,
                                        num_heads=num_heads, qkv_bias=qkv_bias,
                                        qkv_drop=qkv_drop, attn_drop=attn_drop, proj_drop=drop,
                                        pretrained_window_size=_ntuple(img_dim)(pretrained_window_size),
                                        cbp_dim=np.clip(np.prod(input_resolution) // 16, 256, 1024))
        if img_dim == 3:
            self.attn = WindowAttention3D(dim, window_size=self.window_size_tuple,
                                        num_heads=num_heads, qkv_bias=qkv_bias,
                                        qkv_drop=qkv_drop, attn_drop=attn_drop, proj_drop=drop,
                                        pretrained_window_size=_ntuple(img_dim)(pretrained_window_size),
                                        cbp_dim=np.clip(np.prod(input_resolution) // 16, 256, 1024))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = get_norm_layer(dim, norm_layer)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        if shift_size > 0:
            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, *self.input_resolution, 1))  # 1 H W 1

            window_partition_slice = (slice(0, -window_size),
                                    slice(-window_size, -shift_size),
                                    slice(-shift_size, None))
            img_size_slice_list = [deepcopy(window_partition_slice) for _ in self.input_resolution]
            cnt = 0
            for img_size_slice in itertools.product(*img_size_slice_list):
                slice_tuple = tuple((slice(None), *img_size_slice, slice(None)))
                img_mask[slice_tuple] = cnt
                cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size_tuple)
            mask_windows = mask_windows.view(-1, np.prod(self.window_size_tuple))
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, scale_shift_list=[]):

        img_resolution = self.input_resolution
        B, L, C = x.shape
        assert L == np.prod(img_resolution), f"input feature has wrong size {L} != {img_resolution}"
        assert (img_resolution % 2).sum() == 0, f"x size ({img_resolution}) are not even."

        shortcut = x
        x = x.view(B, *img_resolution, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x,
                                   shifts=self.shift_size_tuple,
                                   dims=self.shift_dim_tuple)
        else:
            shifted_x = x

        # partition windows
        # nW*B, window_size, window_size, C
        x_windows = window_partition(shifted_x, self.window_size_tuple)
        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, np.prod(self.window_size_tuple), C)

        # W-MSA/SW-MSA
        # nW*B, window_size*window_size, C
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, *self.window_size_tuple, C)
        shifted_x = window_reverse(attn_windows, self.window_size_tuple, img_resolution)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x,
                           shifts=self.shift_size_tuple, dims=self.shift_dim_tuple)
        else:
            x = shifted_x
        x = x.view(B, L, C)

        norm = self.norm1(x)
        norm = apply_embedding(norm, scale_shift_list)
        x = shortcut + self.drop_path(norm)
        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, img_dim=2):
        super().__init__()
        img_size = np.array(_ntuple(img_dim)(img_size))
        patch_size = np.array(_ntuple(img_dim)(patch_size))
        patches_resolution = img_size // patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = np.prod(patches_resolution)

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        conv_fn = get_conv_nd_fn(img_dim)
        self.proj = conv_fn(in_chans, embed_dim,
                            kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = get_norm_layer(embed_dim, norm_layer)
        else:
            self.norm = None

    def forward(self, x):
        B, C, *img_size_list = x.shape
        # FIXME look at relaxing size constraints
        assert all(self.img_size == img_size_list), \
            f"Input image size ({img_size_list}) doesn't match model ({self.img_size})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    # def flops(self):
    #     Ho, Wo = self.patches_resolution
    #     flops = Ho * Wo * self.embed_dim * self.in_chans * \
    #         (self.patch_size[0] * self.patch_size[1])
    #     if self.norm is not None:
    #         flops += Ho * Wo * self.embed_dim
    #     return flops

class PatchEmbedding(nn.Module):
    def __init__(self, num_patch, in_dim, embed_dim):
        super(PatchEmbedding, self).__init__()

        # Assuming DenseLayer is similar to nn.Linear
        # If there's any spectral normalization or other specifics, you'd need to add those
        self.proj = nn.Linear(in_dim, embed_dim, bias=True)
        self.pos_embed = nn.Embedding(num_patch, embed_dim)

        self.num_patch = num_patch

    def forward(self, patch):
        # PyTorch generally uses [B, C, ...] format
        B, C, _ = patch.shape

        pos = torch.arange(0, self.num_patch).to(patch.device)
        embed = self.proj(patch) + \
            self.pos_embed(pos).unsqueeze(0).repeat(B, 1, 1)

        return embed
    
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: rms
    """

    def __init__(self, input_resolution, dim, norm_layer="rms", img_dim=2):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.img_multiply = (2 ** img_dim)
        binary_seq = self.generate_binary_sequences(img_dim)
        self.slice_tuple_list = self.get_slice_tuple_list(binary_seq)
        self.reduction = nn.Linear(self.img_multiply * dim, 2 * dim, bias=False)
        self.norm = get_norm_layer(2 * dim, norm_layer)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        img_resolution = self.input_resolution
        B, L, C = x.shape
        assert L == np.prod(img_resolution), f"input feature has wrong size {L} != {img_resolution}"
        assert (img_resolution % 2).sum() == 0, f"x size ({img_resolution}) are not even."

        x = x.view(B, *img_resolution, C)
        x = [x[slice_tuple] for slice_tuple in self.slice_tuple_list]
        x = torch.cat(x, -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, self.img_multiply * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)
        return x

    def generate_binary_sequences(self, img_dim):
        return [tuple(seq) for seq in itertools.product([0, 1], repeat=img_dim)]
    
    def get_slice_tuple_list(self, binary_seq):
        slice_tuple_list = []
        all_slice = slice(None, None)
        for slice_start_idx_list in binary_seq:
            img_slice_tuple = [slice(slice_start_idx, None, 2) for slice_start_idx in slice_start_idx_list]
            slice_tuple = tuple((all_slice, *img_slice_tuple, all_slice))
            slice_tuple_list.append(slice_tuple)
        return slice_tuple_list
    
    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops

class PatchExpanding(nn.Module):
    def __init__(self, input_resolution, dim,
                 return_vector=True, dim_scale=2,
                 decode_fn_str="pixel_shuffle", norm_layer="rms", img_dim=2):
        super().__init__()
        support_decode_fn_list = ["upsample", "pixel_shuffle", "conv_transpose"]
        assert decode_fn_str in support_decode_fn_list

        input_resolution = np.array(input_resolution)
        self.input_resolution = input_resolution
        self.dim = dim
        self.return_vector = return_vector
        self.dim_scale = dim_scale
        
        conv_fn = get_conv_nd_fn(img_dim)
        upsample_mode = None
        pixel_shuffle_fn = None
        conv_transpose_fn = None
        decode_layer = None
        if img_dim == 1:
            upsample_mode = "linear"
            pixel_shuffle_fn = PixelShuffle1D
            conv_transpose_fn = nn.ConvTranspose1d
        elif img_dim == 2:
            upsample_mode = "bilinear"
            pixel_shuffle_fn = nn.PixelShuffle
            conv_transpose_fn = nn.ConvTranspose2d
        elif img_dim == 3:
            upsample_mode = "trilinear"
            pixel_shuffle_fn = PixelShuffle3D
            conv_transpose_fn = nn.ConvTranspose3d

        middle_dim = 0
        decode_layer_list = []
        if decode_fn_str == "upsample":
            upsample_conv = conv_fn(dim, dim // 2,
                                    kernel_size=1, padding=0, bias=False)
            upsample_layer = nn.Sequential(
                upsample_conv,
                nn.Upsample(scale_factor=dim_scale, mode=upsample_mode)
            )
            middle_dim += dim // 2
            decode_layer_list.append(upsample_layer)
        elif decode_fn_str == "pixel_shuffle":
            pixel_conv = conv_fn(dim, dim * (dim_scale ** img_dim) // 2,
                                    kernel_size=1, padding=0, bias=False)
            if dim_scale == 1:
                pixel_shuffle = pixel_conv
            else:
                pixel_shuffle = nn.Sequential(
                                pixel_conv,
                                pixel_shuffle_fn(upscale_factor=dim_scale)
                )
            decode_layer_list.append(pixel_shuffle)
            middle_dim += dim // 2
        elif decode_fn_str == "conv_transpose":
            conv_transpose_layer = conv_transpose_fn(dim, dim // 2, stride=dim_scale,
                                             **conv_transpose_kwarg_dict)
            decode_layer_list.append(conv_transpose_layer)
            middle_dim += dim // 2
        self.concat_conv = ConvBlockND(middle_dim, dim // 2, kernel_size=3, stride=1, padding=1, norm=get_rms_norm_nd(img_dim), act="silu",
                                       bias=False, dropout_proba=0.0, image_shape=None, img_dim=img_dim)
        self.decode_layer = nn.ModuleList(decode_layer_list)
        self.norm_layer = get_norm_layer(dim // 2, norm_layer)

    def forward(self, x):
        img_resolution = self.input_resolution
        permute_tuple = tuple(range(2, 2 + len(img_resolution)))
        B, L, C = x.shape
        assert L == np.prod(img_resolution), f"input feature has wrong size {L} != {img_resolution}"
        assert (img_resolution % 2).sum() == 0, f"x size ({img_resolution}) are not even."
        x = x.permute(0, 2, 1).view(B, C, *img_resolution)

        x = [decode_layer(x) for decode_layer in self.decode_layer]
        x = torch.cat(x, dim=1)
        
        x = x.permute(0, *permute_tuple, 1).view(B, -1, self.dim // 2)
        x = self.norm_layer(x)
        if not self.return_vector:
            x = x.permute(0, 2, 1).view(B, self.dim // 2, *(img_resolution * self.dim_scale))
        return x
    
class BasicLayerV1(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., qkv_drop=0., attn_drop=0.,
                 drop_path=0., norm_layer="rms", act_layer=DEFAULT_ACT, downsample=None, upsample=None, decode_fn_str="pixel_shuffle",
                 use_checkpoint=False, pretrained_window_size=0, emb_dim_list=[], emb_type_list=[], img_dim=2):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.emb_block_list = get_emb_block_list(act_layer, emb_dim_list, emb_type_list, dim)
        self.emb_type_list = emb_type_list

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, qkv_drop=qkv_drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer, act_layer=act_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution,
                                         dim=dim, norm_layer=norm_layer, img_dim=img_dim)
        else:
            self.downsample = None
        if upsample is not None:
            self.upsample = upsample(input_resolution,
                                     dim=dim, norm_layer=norm_layer,
                                     decode_fn_str=decode_fn_str, img_dim=img_dim)
        else:
            self.upsample = None
        self._init_respostnorm()

    def forward(self, x, *emb_args):
        scale_shift_list = get_scale_shift_list(self.emb_block_list, self.emb_type_list, emb_args)
        for blk in self.blocks:
            x = process_checkpoint_block(self.use_checkpoint, blk, x, scale_shift_list)
        if self.downsample is not None:
            x = process_checkpoint_block(self.use_checkpoint, self.downsample, x)
        elif self.upsample is not None:
            x = process_checkpoint_block(self.use_checkpoint, self.upsample, x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            check_hasattr_and_init(blk.norm1, "weight", 1)
            check_hasattr_and_init(blk.norm1, "bias", 0)
            check_hasattr_and_init(blk.norm2, "weight", 1)
            check_hasattr_and_init(blk.norm2, "bias", 0)

class BasicLayerV2(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., qkv_drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, act_layer=DEFAULT_ACT, downsample=None, upsample=None, decode_fn_str="pixel_shuffle",
                 use_checkpoint=False, pretrained_window_size=0, emb_dim_list=[], emb_type_list=[], img_dim=2):

        super().__init__()

        input_resolution = np.array(input_resolution)
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution,
                                         dim=dim, norm_layer=norm_layer, img_dim=img_dim)
            dim *= 2
            num_heads *= 2
            window_size = max(window_size // 2, 2)
            input_resolution = input_resolution // 2
        else:
            self.downsample = None
        if upsample is not None:
            self.upsample = upsample(input_resolution,
                                     dim=dim, norm_layer=norm_layer,
                                     decode_fn_str=decode_fn_str, img_dim=img_dim)
            dim //= 2
            num_heads = max(num_heads // 2, 1)
            window_size *= 2
            input_resolution = input_resolution * 2
        else:
            self.upsample = None

        self.emb_block_list = get_emb_block_list(act_layer, emb_dim_list, emb_type_list, dim)
        self.emb_type_list = emb_type_list
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, qkv_drop=qkv_drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer, act_layer=act_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])
        self._init_respostnorm()

    def forward(self, x, *emb_args):
        scale_shift_list = get_scale_shift_list(self.emb_block_list, self.emb_type_list, emb_args)
        if self.downsample is not None:
            x = process_checkpoint_block(self.use_checkpoint, self.downsample, x)
        elif self.upsample is not None:
            x = process_checkpoint_block(self.use_checkpoint, self.upsample, x)
        for blk in self.blocks:
            x = process_checkpoint_block(self.use_checkpoint, blk, x, scale_shift_list)
        return x
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            check_hasattr_and_init(blk.norm1, "weight", 1)
            check_hasattr_and_init(blk.norm1, "bias", 0)
            check_hasattr_and_init(blk.norm2, "weight", 1)
            check_hasattr_and_init(blk.norm2, "bias", 0)

class SkipEncodeLayer(nn.Module):
    def __init__(self, dim, skip_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., qkv_drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, act_layer=DEFAULT_ACT, 
                 downsample=None, use_checkpoint=False, pretrained_window_size=0,
                 emb_dim_list=[], emb_type_list=[], img_dim=2):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        before_depth = depth // 2
        after_depth = depth - before_depth

        self.blocks_before_skip = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, qkv_drop=qkv_drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(
                                     drop_path, list) else drop_path,
                                 norm_layer=norm_layer, act_layer=act_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(before_depth)])
        self.mlp_after_skip = Mlp(in_features=dim + skip_dim,
                                  out_features=dim,
                                  act_layer=act_layer, drop=drop)        
        if downsample is not None:
            self.downsample = downsample(input_resolution,
                                         dim=dim, norm_layer=norm_layer, img_dim=img_dim)
            dim *= 2
            num_heads *= 2
            window_size = max(window_size // 2, 2)
            input_resolution = input_resolution // 2
        else:
            self.downsample = None

        self.emb_block_list = get_emb_block_list(act_layer, emb_dim_list, emb_type_list, dim)
        self.emb_type_list = emb_type_list
        # build blocks

        self.blocks_after_skip = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, qkv_drop=qkv_drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(
                                     drop_path, list) else drop_path,
                                 norm_layer=norm_layer, act_layer=act_layer,
                                 pretrained_window_size=pretrained_window_size,
                                 img_dim=img_dim)
            for i in range(after_depth)])
        self._init_respostnorm()

    def forward(self, x, skip, *emb_args):
        scale_shift_list = get_scale_shift_list(self.emb_block_list, self.emb_type_list, emb_args)
        for blk in self.blocks_before_skip:
            x = process_checkpoint_block(self.use_checkpoint, blk, x, scale_shift_list)
        x = torch.cat([x, skip], dim=-1)
        x = self.mlp_after_skip(x)
        if self.downsample is not None:
            x = process_checkpoint_block(self.use_checkpoint, self.downsample, x)
        for blk in self.blocks_after_skip:
            x = process_checkpoint_block(self.use_checkpoint, blk, x, scale_shift_list)
        return x
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks_before_skip + self.blocks_after_skip:
            check_hasattr_and_init(blk.norm1, "weight", 1)
            check_hasattr_and_init(blk.norm1, "bias", 0)
            check_hasattr_and_init(blk.norm2, "weight", 1)
            check_hasattr_and_init(blk.norm2, "bias", 0)

class BasicDecodeLayer(nn.Module):
    def __init__(self, dim, skip_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., qkv_drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, act_layer=DEFAULT_ACT, 
                 upsample=None, decode_fn_str="pixel_shuffle",
                 use_checkpoint=False, pretrained_window_size=0,
                 emb_dim_list=[], emb_type_list=[], img_dim=2):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        before_depth = depth // 2
        after_depth = depth - before_depth

        self.blocks_before_skip = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, qkv_drop=qkv_drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(
                                     drop_path, list) else drop_path,
                                 norm_layer=norm_layer, act_layer=act_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(before_depth)])
        if upsample is not None:
            self.upsample = upsample(input_resolution,
                                    dim=dim, norm_layer=norm_layer,
                                    decode_fn_str=decode_fn_str, img_dim=img_dim)
            dim //= 2
            num_heads = max(num_heads // 2, 1)
            window_size *= 2
            input_resolution = input_resolution * 2
        else:
            self.upsample = None

        self.emb_block_list = get_emb_block_list(act_layer, emb_dim_list, emb_type_list, dim)
        self.emb_type_list = emb_type_list
        # build blocks
        self.mlp_after_skip = Mlp(in_features=dim + skip_dim,
                                  out_features=dim,
                                  act_layer=act_layer, drop=drop)
        self.blocks_after_skip = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, qkv_drop=qkv_drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(
                                     drop_path, list) else drop_path,
                                 norm_layer=norm_layer, act_layer=act_layer,
                                 pretrained_window_size=pretrained_window_size,
                                 img_dim=img_dim)
            for i in range(after_depth)])
        self._init_respostnorm()

    def forward(self, x, skip, *emb_args):
        scale_shift_list = get_scale_shift_list(self.emb_block_list, self.emb_type_list, emb_args)
        for blk in self.blocks_before_skip:
            x = process_checkpoint_block(self.use_checkpoint, blk, x, scale_shift_list)
        if self.upsample is not None:
            x = process_checkpoint_block(self.use_checkpoint, self.upsample, x)
        x = torch.cat([x, skip], dim=-1)
        x = self.mlp_after_skip(x)
        for blk in self.blocks_after_skip:
            x = process_checkpoint_block(self.use_checkpoint, blk, x, scale_shift_list)
        return x
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks_before_skip + self.blocks_after_skip:
            check_hasattr_and_init(blk.norm1, "weight", 1)
            check_hasattr_and_init(blk.norm1, "bias", 0)
            check_hasattr_and_init(blk.norm2, "weight", 1)
            check_hasattr_and_init(blk.norm2, "bias", 0)
class AttentionPool1d(nn.Module):
    def __init__(self, sequence_length: int, embed_dim: int,
                 num_heads: int, output_dim: int = None,
                 channel_first: bool = True):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(sequence_length + 1,
                                                             embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.channel_first = channel_first

    def forward(self, x):
        if self.channel_first:
            x = x.permute(2, 0, 1)
        else:
            x = x.permute(1, 0, 2)  # NCL -> LNC

        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (L+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (L+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias,
                                    self.k_proj.bias,
                                    self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)

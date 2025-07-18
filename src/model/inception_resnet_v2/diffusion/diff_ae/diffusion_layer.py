from collections import namedtuple
from packaging import version
from functools import wraps
from functools import partial
import math
import numpy as np
from typing import NamedTuple, Optional, Any, Iterable
from dataclasses import dataclass, field
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat
from src.model.inception_resnet_v2.common_module.cbam import CBAM
from src.model.inception_resnet_v2.common_module.layers import get_act, DEFAULT_ACT, INPLACE
from src.model.inception_resnet_v2.common_module.layers import ConcatBlock
from .nn import zero_module, conv_nd
from src.model.inception_resnet_v2.diffusion.diff_ae.flash_attn import FlashMultiheadAttention
conv_transpose_kwarg_dict = {
    "kernel_size":3,
    "padding":1,
    "output_padding":1,
    "bias":True
}

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

def _z_normalize(x, target_dim_tuple, eps=1e-5):
    x_mean = x.mean(dim=target_dim_tuple, keepdim=True)
    x_var = x.var(dim=target_dim_tuple, correction=0, keepdim=True)
    x_std = torch.sqrt(x_var + eps)
    x_normalized = (x - x_mean) / x_std
    return x_normalized

def feature_z_normalize(x, eps=1e-5):
    target_dim_tuple = tuple(range(1, x.ndim))
    return _z_normalize(x, target_dim_tuple, eps=eps)

def z_normalize(x, eps=1e-5):
    target_dim_tuple = tuple(range(2, x.ndim))
    return _z_normalize(x, target_dim_tuple, eps=eps)

class EmbedSequential(nn.Sequential):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    def forward(self, x, *emb_arg):
        for layer in self:
            x = layer(x, *emb_arg)
        return x

@dataclass
class Return:
    pred: Optional[torch.Tensor] = None
    pred_ancher: Optional[torch.Tensor] = None
    encoded_feature: Optional[torch.Tensor] = None
    seg_pred: Optional[torch.Tensor] = None
    class_pred: Optional[torch.Tensor] = None
    recon_pred: Optional[torch.Tensor] = None
    validity_pred: Optional[torch.Tensor] = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)

class WrapGroupNorm(nn.GroupNorm):
    
    def __init__(self, num_channels, *args, **kwargs):
        super().__init__(num_channels=num_channels,*args, **kwargs)
        
class LayerNorm(nn.Module):
    def __init__(self, dim, bias = False):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1)) if bias else None

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g + default(self.b, 0)
print_once = once(print)

# main class
AttentionConfig = namedtuple('AttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False,
        scale = None
    ):
        super().__init__()
        self.dropout = dropout
        self.scale = scale
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = AttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        if exists(self.scale):
            default_scale = q.shape[-1]
            q = q * (self.scale / default_scale)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p = self.dropout if self.training else 0.
            )

        return out

    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        if self.flash:
            return self.flash_attn(q, k, v)

        scale = default(self.scale, q.shape[-1] ** -0.5)

        # similarity

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out
    
class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dim_head=32, num_mem_kv=4, use_checkpoint=False):
        super().__init__()
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        if dim_head is None:
            dim_head = dim
        self.hidden_dim = dim_head * num_heads
        self.scale = (dim_head / num_heads) ** -0.5
        self.mem_kv = nn.Parameter(torch.randn(2, num_heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, self.hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(self.hidden_dim, dim, 1),
            LayerNorm(dim)
        )
        
    def forward(self, x):
        if self.use_checkpoint:
            return checkpoint(self._forward_impl, x,
                              use_reentrant=False)
        else:
            return self._forward_impl(x)
        
    def _forward_impl(self, x):
        b, c, h, w = x.shape

        x = z_normalize(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.num_heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.num_heads, x = h, y = w)
        return self.to_out(out) + x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, dim_head=32, num_mem_kv=4, flash=False, use_checkpoint=False):
        super().__init__()
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        if dim_head is None:
            dim_head = dim

        self.hidden_dim = dim_head * num_heads
        self.attend = Attend(flash = flash)
        self.mem_kv = nn.Parameter(torch.randn(2, num_heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, self.hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(self.hidden_dim, dim, 1)

    def forward(self, x):
        if self.use_checkpoint:
            return checkpoint(self._forward_impl, x,
                              use_reentrant=False)
        else:
            return self._forward_impl(x)
        
    def _forward_impl(self, x):
        b, c, h, w = x.shape

        x = z_normalize(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.num_heads), qkv)
        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out) + x

class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch,
                                                                       dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale,
            k * scale)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight,
                      v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

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
        use_new_attention_order=False,
        use_flash_attention=False,
        norm_layer="rms"
    ): 
        super().__init__()
        assert norm_layer in ["rms", "group"]
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        if norm_layer == "rms":
            self.norm = RMSNorm(channels, img_dim=1)
        elif norm_layer== "group":
            self.norm = GroupNorm32(channels)
        
        self.use_flash_attention = use_flash_attention
        if use_flash_attention:
            self.qkv = None
            self.attention = FlashMultiheadAttention(dim=channels, num_heads=self.num_heads, causal=False)
        else:
            self.qkv = conv_nd(1, channels, channels * 3, 1)
            if use_new_attention_order:
                # split qkv before split heads
                self.attention = QKVAttention(self.num_heads)
            else:
                # split heads before split qkv
                self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, *args):
        if self.use_checkpoint:
            return checkpoint(self._forward_impl, x, *args,
                              use_reentrant=False)
        else:
            return self._forward_impl(x, *args)

    def _forward_impl(self, x, *args):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        h = x
        x = self.norm(x)
        if self.use_flash_attention:
            x = rearrange(x, 'b c n -> b n c')
            x = self.attention(x)
            x = rearrange(x, 'b n c -> b c n')
        else:
            qkv = self.qkv(x)
            x = self.attention(qkv)
        x = self.proj_out(x)
        return (x + h).reshape(b, c, *spatial)
    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
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

def get_scale_shift_list(emb_block_list, emb_type_list, emb_args, img_dim):
    unsqueeze_dim = tuple(1 for _ in range(img_dim))
    scale_shift_list = []
    for emb_block, emb_type, emb in zip(emb_block_list, emb_type_list, emb_args):
        if (emb_block is None) or (emb is None):
            scale_shift_list.append([None, None])
        else:
            emb = emb_block(emb)
            emb_shape = emb.shape
            emb = emb.view(*emb_shape, *unsqueeze_dim)
            if emb_type == "cond":
                scale = emb
                shift = torch.zeros_like(emb)
            else:
                scale, shift = emb.chunk(2, dim=1)
            scale_shift_list.append([scale, shift])
    return scale_shift_list

def apply_embedding(x, scale_shift_list, scale_bias=1):
    for scale, shift in scale_shift_list:
        if scale is not None:
            x = x * (scale_bias + scale) + shift
    return x

class GroupNorm32(nn.GroupNorm):
    def __init__(self, num_channels: int, *args, **kwargs) -> None:
        if num_channels % 3 == 0:
            num_groups = min(24, num_channels)
        elif num_channels % 5 == 0:
            num_groups = min(40, num_channels)
        elif num_channels % 7 == 0:
            num_groups = min(28, num_channels)
        else:
            num_groups = min(32, num_channels)
        super().__init__(num_groups, num_channels, *args, **kwargs)

def get_conv_nd_fn(img_dim):
    assert img_dim in [1, 2, 3]
    conv_fn = None
    if img_dim == 1:
        conv_fn = nn.Conv1d
    elif img_dim == 2:
        conv_fn = nn.Conv2d
    elif img_dim == 3:
        conv_fn = nn.Conv3d
    return conv_fn

class BaseBlockND(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding='same', dilation=1, norm=nn.LayerNorm, groups=1, act=DEFAULT_ACT, bias=False,
                 dropout_proba=0.0, attn_layer=nn.Identity(), image_shape=None, img_dim=2, separable=False):
        super().__init__()
        conv_fn = get_conv_nd_fn(img_dim)
        if kernel_size == 1:
            separable = False
        if separable:
            self.conv = conv_fn(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                                groups=groups, bias=bias)
        else:
            self.conv = nn.Sequential(
                    conv_fn(in_channels=in_channels, out_channels=in_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                                groups=groups, bias=bias),
                    conv_fn(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=1, stride=1, padding=0,
                                groups=groups, bias=bias)
            )
        
        self.attn_layer = attn_layer
        if (not bias) and (norm is not None):
            if image_shape is None:
                image_shape = out_channels
            else:
                image_shape = (out_channels, *image_shape)
            self.norm_layer = norm(image_shape)
        else:
            self.norm_layer = nn.Identity()
        self.act_layer = get_act(act)
        self.dropout_layer = nn.Dropout(p=dropout_proba, inplace=INPLACE)
    
    def forward(self, x, scale_shift_list=[]):
        x = self.conv(x)
        x = apply_embedding(x, scale_shift_list)
        x = self.attn_layer(x)
        x = self.norm_layer(x)
        x = self.act_layer(x)
        x = self.dropout_layer(x)
        return x

def get_attn_layer(out_channels, attn_info, use_checkpoint):
    attn_layer = None
    if attn_info is None:
        attn_layer = nn.Identity()
    elif attn_info["full_attn"] is None:
        attn_layer = nn.Identity()
    elif attn_info["full_attn"] == []:
        attn_layer = nn.Identity()
    elif attn_info["full_attn"] == True:
        attn_layer = AttentionBlock(out_channels, attn_info["num_heads"], use_checkpoint=use_checkpoint)
    elif attn_info["full_attn"] == "flash-attn":
        attn_layer = AttentionBlock(out_channels, attn_info["num_heads"], use_checkpoint=use_checkpoint, use_flash_attention=True)
    else:
        attn_layer = LinearAttention(out_channels, attn_info["num_heads"], attn_info["dim_head"], use_checkpoint=use_checkpoint)
    return attn_layer

#attn_info.keys = ["emb_type_list", "num_heads", "full_attn"]
class ResNetBlockND(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding='same', dilation=1,
                 norm="batch", groups=1, act=DEFAULT_ACT, bias=False, dropout_proba=0.0,
                 emb_dim_list=[], emb_type_list=[], attn_info=None, use_checkpoint=False, image_shape=None, 
                 img_dim=2, separable=False):
        super().__init__()
        conv_fn = get_conv_nd_fn(img_dim)
        self.img_dim = img_dim
        self.attn_info = attn_info
        self.use_checkpoint = use_checkpoint

        emb_block_list = []
        for emb_dim, emb_type in zip(emb_dim_list, emb_type_list):
            if emb_type == "seq":
                emb_block = nn.Sequential(
                                            get_act(act),
                                            nn.Linear(emb_dim, out_channels * 2)
                                        )
            elif emb_type == "cond":
                emb_block = nn.Sequential(
                                            get_act(act),
                                            nn.Linear(emb_dim, out_channels)
                                        )
            elif emb_type == "nd":
                emb_block = BaseBlockND(emb_dim, out_channels * 2, kernel_size=kernel_size,
                                        stride=1, padding=padding, dilation=1, norm=norm,
                                        groups=groups, act=act, bias=bias, dropout_proba=0.0, img_dim=img_dim,
                                        separable=separable)
            elif emb_type == "none" or (emb_type is None):
                emb_block = None
            else:
                raise NotImplementedError(f"emb_type must be in [seq, cond, nd]")
            emb_block_list.append(emb_block)
        self.emb_block_list = nn.ModuleList(emb_block_list)
        self.emb_type_list = emb_type_list

        if in_channels != out_channels or stride != 1:
            self.residiual_conv = conv_fn(in_channels, out_channels, 1, stride, bias=False)
        else:
            self.residiual_conv = nn.Identity()

        attn_layer = get_attn_layer(out_channels, attn_info, use_checkpoint)
        self.block_1 = BaseBlockND(in_channels, out_channels, kernel_size,
                                    stride=1, padding=padding, dilation=dilation, norm=norm, groups=groups, act=act, bias=bias, dropout_proba=dropout_proba,
                                    image_shape=image_shape, attn_layer=nn.Identity(), img_dim=img_dim, separable=separable)
        self.block_2 = BaseBlockND(out_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, norm=norm, groups=groups, act=act, bias=bias, dropout_proba=dropout_proba,
                                    image_shape=image_shape, attn_layer=attn_layer, img_dim=img_dim, separable=separable)
        
    def forward(self, x, *args):
        if self.use_checkpoint:
            conv_output = checkpoint(self._forward_conv, x, *args,
                              use_reentrant=False)
        else:
            conv_output = self._forward_conv(x, *args)
        return conv_output
    
    def _forward_conv(self, x, *emb_args):
        skip_x = x
        scale_shift_list = get_scale_shift_list(self.emb_block_list, self.emb_type_list,
                                                emb_args, self.img_dim)
        x = self.block_1(x)
        x = self.block_2(x, scale_shift_list)
        x = x + self.residiual_conv(skip_x)
        return x
    
class ConvBlockND(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding='same', dilation=1,
                 norm="batch", groups=1, act=DEFAULT_ACT, bias=False, dropout_proba=0.0,
                 emb_dim_list=[], emb_type_list=[], attn_info=None, use_checkpoint=False,
                 image_shape=None, img_dim=2, separable=False):
        super().__init__()
        self.img_dim = img_dim
        self.attn_info = attn_info
        self.use_checkpoint = use_checkpoint

        if emb_dim_list is None:
            emb_dim_list = []
        if emb_type_list is None:
            emb_type_list = []
        emb_block_list = []
        for emb_dim, emb_type in zip(emb_dim_list, emb_type_list):
            if emb_type == "seq":
                emb_block = nn.Sequential(
                                            get_act(act),
                                            nn.Linear(emb_dim, out_channels * 2)
                                        )
            elif emb_type == "cond":
                emb_block = nn.Sequential(
                                            get_act(act),
                                            nn.Linear(emb_dim, out_channels)
                                        )
            elif emb_type == "nd":
                emb_block = BaseBlockND(emb_dim, out_channels * 2, kernel_size,
                                        stride=1, padding=padding, dilation=dilation, norm=norm, groups=groups,
                                        act=act, bias=bias, dropout_proba=0.0, img_dim=img_dim)
            elif emb_type == "none" or (emb_type is None):
                emb_block = None
            else:
                raise NotImplementedError(f"emb_type must be in [seq, cond, nd]")
            emb_block_list.append(emb_block)
        self.emb_block_list = nn.ModuleList(emb_block_list)
        self.emb_type_list = emb_type_list

        attn_layer = get_attn_layer(out_channels, attn_info, use_checkpoint)
        self.block_1 = BaseBlockND(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, norm=norm, groups=groups,
                                    act=act, bias=bias, dropout_proba=dropout_proba, attn_layer=attn_layer,
                                    image_shape=image_shape, img_dim=img_dim, separable=separable)

    def forward(self, x, *args):
        if self.use_checkpoint:
            conv_output = checkpoint(self._forward_conv, x, *args,
                              use_reentrant=False)
        else:
            conv_output = self._forward_conv(x, *args)
        return conv_output
    
    def _forward_conv(self, x, *emb_args):
        scale_shift_list = get_scale_shift_list(self.emb_block_list, self.emb_type_list,
                                                emb_args, self.img_dim)
        x = self.block_1(x, scale_shift_list)
        return x

class ResNetBlockNDSkip(ResNetBlockND):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, skip, *emb_args):
        if self.use_checkpoint:
            return checkpoint(self._forward_impl, x, skip, *emb_args,
                              use_reentrant=False)
        else:
            return self._forward_impl(x, skip, *emb_args)

    def _forward_impl(self, x, skip, *emb_args):
        x = torch.cat([x, skip], dim=1)
        x = super().forward(x, *emb_args)
        return x
    
class ConvBlockNDSkip(ConvBlockND):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, skip, *emb_args):
        if self.use_checkpoint:
            return checkpoint(self._forward_impl, x, skip, *emb_args,
                              use_reentrant=False)
        else:
            return self._forward_impl(x, skip, *emb_args)

    def _forward_impl(self, x, skip, *emb_args):
        x = torch.cat([x, skip], dim=1)
        x = super().forward(x, *emb_args)
        return x

class Inception_Resnet_BlockND(nn.Module):
    def __init__(self, in_channels, scale, block_type, block_size=16,
                 include_cbam=True, norm="batch", act=DEFAULT_ACT, dropout_proba=0.0,
                 emb_dim_list=None, attn_info=None, use_checkpoint=False):
        super().__init__()
        assert emb_dim_list is not None, f"You need to set emb_dim_list. current emb_dim_list: {emb_dim_list}"
        self.include_cbam = include_cbam
        self.scale = scale
        common_kwarg_dict = {
            "norm": norm,
            "act": act,
            "dropout_proba": dropout_proba,
            "emb_dim_list": emb_dim_list,
            "attn_info": attn_info,
            "use_checkpoint": use_checkpoint
        }

        if block_type == 'block35':
            branch_0 = ConvBlockND(in_channels, block_size * 2, 1,
                                   **common_kwarg_dict)
            branch_1 = MultiInputSequential(
                ConvBlockND(in_channels, block_size * 2, 1,
                            **common_kwarg_dict),
                ConvBlockND(block_size * 2, block_size * 2, 3,
                            **common_kwarg_dict)
            )
            branch_2 = MultiInputSequential(
                ConvBlockND(in_channels, block_size * 2, 1,
                            **common_kwarg_dict),
                ConvBlockND(block_size * 2, block_size * 3, 3,
                            **common_kwarg_dict),
                ConvBlockND(block_size * 3, block_size * 4, 3,
                            **common_kwarg_dict)
            )
            mixed_channel = block_size * 8
            # block_size * (2 + 2 + 4) => block_size * 10
            branches = [branch_0, branch_1, branch_2]
        elif block_type == 'block17':
            branch_0 = ConvBlockND(in_channels, block_size * 12, 1,
                                   **common_kwarg_dict)
            branch_1 = MultiInputSequential(
                ConvBlockND(in_channels, block_size * 8, 1,
                            **common_kwarg_dict),
                ConvBlockND(block_size * 8, block_size * 10, [1, 7],
                            **common_kwarg_dict),
                ConvBlockND(block_size * 10, block_size * 12, [7, 1],
                            **common_kwarg_dict)
            )
            mixed_channel = block_size * 24
            branches = [branch_0, branch_1]
        elif block_type == 'block8':
            branch_0 = ConvBlockND(in_channels, block_size * 12, 1,
                                   **common_kwarg_dict)
            branch_1 = MultiInputSequential(
                ConvBlockND(in_channels, block_size * 12, 1,
                            **common_kwarg_dict),
                ConvBlockND(block_size * 12, block_size * 14, [1, 3],
                            **common_kwarg_dict),
                ConvBlockND(block_size * 14, block_size * 16, [3, 1],
                            **common_kwarg_dict)
            )
            mixed_channel = block_size * 28
            branches = [branch_0, branch_1]
        else:
            raise ValueError('Unknown Inception-ResNet block type. '
                             'Expects "block35", "block17" or "block8", '
                             'but got: ' + str(block_type))
        self.mixed = ConcatBlock(branches)
        # TBD: Name?
        self.up = ConvBlockND(mixed_channel, in_channels, 1,
                              bias=True, norm=norm, act=None,
                              emb_dim_list=emb_dim_list, attn_info=attn_info,
                              use_checkpoint=use_checkpoint)
        if self.include_cbam:
            self.cbam = CBAM(gate_channels=in_channels,
                             reduction_ratio=16)
        self.act = get_act(act)

    def forward(self, x, *args):
        mixed = self.mixed(x, *args)
        up = self.up(mixed, *args)
        if self.include_cbam:
            up = self.cbam(up)
        residual_add = x + up * self.scale
        act = self.act(residual_add)
        return act

class PixelShuffle1D(nn.Module):
    def __init__(self, upscale_factor):
        '''
        :param scale: upsample scale
        '''
        super().__init__()

        if (not isinstance(upscale_factor, int)) or (not isinstance(upscale_factor, float)):
            upscale_factor = upscale_factor[0]

        self.scale = upscale_factor
        self.scale_size = self.scale

    def forward(self, input):
        batch_size, channels, in_seq_len = input.size()
        nOut = channels // self.scale_size

        out_seq_len = in_seq_len * self.scale
        input_view = input.view(batch_size, nOut, self.scale, out_seq_len)
        output = input_view.permute(0, 1, 3, 2)
        output = output.contiguous()
        return output

class PixelShuffle3D(nn.Module):
    '''
    Source: https://github.com/kuoweilai/pixelshuffle3d/blob/master/pixelshuffle3d.py
    This class is a 3d version of pixelshuffle.
    '''

    def __init__(self, upscale_factor):
        '''
        :param scale: upsample scale
        '''
        super().__init__()

        if isinstance(upscale_factor, int):
            upscale_factor = (upscale_factor, upscale_factor, upscale_factor)
        self.scale = upscale_factor
        self.scale_size = np.prod(self.scale)

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale_size

        out_depth = in_depth * self.scale[0]
        out_height = in_height * self.scale[1]
        out_width = in_width * self.scale[2]

        input_view = input.view(batch_size, nOut, *self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4)
        output = output.contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)

class MultiDecoderND(nn.Module):
    def __init__(self, in_channels, out_channels,
                 norm="layer", act=DEFAULT_ACT, kernel_size=2,
                 emb_dim_list=None, emb_type_list=None, attn_info=None, use_checkpoint=False,
                 img_dim=2):
        super().__init__()

        conv_fn = get_conv_nd_fn(img_dim)
        self.use_checkpoint = use_checkpoint
        if isinstance(kernel_size, int):
            kernel_size = tuple(kernel_size for _ in range(img_dim))

        conv_before_pixel_shuffle = conv_fn(in_channels=in_channels,
                                              out_channels=in_channels *
                                              np.prod(kernel_size),
                                              kernel_size=1)
        pixel_shuffle_fn = None
        upsample_mode = None
        if img_dim == 1:
            pixel_shuffle_fn = PixelShuffle1D
            upsample_mode = "linear"
        elif img_dim == 2:
            pixel_shuffle_fn = nn.PixelShuffle
            upsample_mode = "bilinear"
        elif img_dim == 3:
            pixel_shuffle_fn = PixelShuffle3D
            upsample_mode = "trilinear"
            
        conv_after_pixel_shuffle = conv_fn(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=1)
        self.pixel_shuffle = nn.Sequential(
            conv_before_pixel_shuffle,
            pixel_shuffle_fn(upscale_factor=kernel_size[0]),
            conv_after_pixel_shuffle
        )
        upsample_layer = nn.Upsample(scale_factor=kernel_size,
                                     mode=upsample_mode)
        conv_after_upsample = conv_fn(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=1)
        self.upsample = nn.Sequential(
            upsample_layer,
            conv_after_upsample
        )
        self.concat_conv = ConvBlockND(in_channels=out_channels * 2,
                                       out_channels=out_channels, kernel_size=3,
                                       stride=1, padding=1, norm=norm, act=act,
                                       emb_dim_list=emb_dim_list, emb_type_list=emb_type_list, attn_info=attn_info,
                                       use_checkpoint=use_checkpoint, img_dim=img_dim)

    def forward(self, x, *args):
        if self.use_checkpoint:
            return checkpoint(self._forward_impl, x, *args,
                              use_reentrant=False)
        else:
            return self._forward_impl(x, *args)
        
    def _forward_impl(self, x, *args):
        pixel_shuffle = self.pixel_shuffle(x)
        upsample = self.upsample(x)
        out = torch.cat([pixel_shuffle, upsample], dim=1)
        out = self.concat_conv(out, *args)
        return out

class MultiDecoderND_V2(nn.Module):
    def __init__(self, in_channels, out_channels,
                 norm="layer", act=DEFAULT_ACT, kernel_size=2, dropout_proba=0.0,
                 emb_dim_list=[], emb_type_list=[], attn_info=None, use_checkpoint=False,
                 image_shape=None, decode_fn_str_list=["pixel_shuffle"], img_dim=2, use_residual_conv=True):
        super().__init__()

        support_decode_fn_list = ["upsample", "pixel_shuffle", "conv_transpose"]
        assert img_dim in [1, 2, 3]
        
        for decode_fn_str in decode_fn_str_list:
            assert decode_fn_str in support_decode_fn_list

        conv_fn = get_conv_nd_fn(img_dim)
        self.use_checkpoint = use_checkpoint
        if image_shape is not None:
            image_shape = image_shape * 2
        conv_common_kwarg_dict = {
            "kernel_size": 3, "stride": 1, "padding": 1,
            "dropout_proba": dropout_proba,
            "norm": norm, "act": act,
            "emb_dim_list": emb_dim_list,
            "emb_type_list": emb_type_list,
            "attn_info": attn_info,
            "use_checkpoint": use_checkpoint,
            "image_shape": image_shape,
            "img_dim": img_dim
        }
        if isinstance(kernel_size, int):
            kernel_size = tuple(kernel_size for _ in range(img_dim))

        decode_middle_channel = 0
        decode_layer_list = []
        upsample_mode = None
        pixel_shuffle_fn = None
        conv_transpose_fn = None

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
            

        if "upsample" in decode_fn_str_list:
            upsample_layer = nn.Upsample(scale_factor=kernel_size, mode=upsample_mode)
            decode_layer_list.append(upsample_layer)
            decode_middle_channel += in_channels

        if "pixel_shuffle" in decode_fn_str_list:
            conv_before_pixel_shuffle = conv_fn(in_channels=in_channels,
                                        out_channels=in_channels *
                                        np.prod(kernel_size),
                                        kernel_size=1)
            pixel_shuffle = nn.Sequential(
                conv_before_pixel_shuffle,
                pixel_shuffle_fn(upscale_factor=kernel_size[0])
            )
            decode_layer_list.append(pixel_shuffle)
            decode_middle_channel += in_channels

        if "conv_transpose" in decode_fn_str_list:
            conv_transpose = conv_transpose_fn(in_channels, in_channels,
                                                    stride=kernel_size,
                                                    **conv_transpose_kwarg_dict)
            decode_layer_list.append(conv_transpose)
            decode_middle_channel += in_channels

        self.decode_layer_list = nn.ModuleList(decode_layer_list)
        if use_residual_conv:
            conv_block = ResNetBlockND
        else:
            conv_block = ConvBlockND
        self.concat_conv = conv_block(in_channels=decode_middle_channel,
                                      out_channels=out_channels, **conv_common_kwarg_dict)
    def forward(self, x, *args):
        if self.use_checkpoint:
            return checkpoint(self._forward_impl, x, *args,
                              use_reentrant=False)
        else:
            return self._forward_impl(x, *args)
        
    def _forward_impl(self, x, *args):

        decode_output_list = [decode_layer(x) for decode_layer in self.decode_layer_list]
        decode_output = torch.cat(decode_output_list, dim=1)
        decode_output = self.concat_conv(decode_output, *args)
        return decode_output

class MultiDecoderND_V3(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels,
                 norm="layer", act=DEFAULT_ACT, kernel_size=2, dropout_proba=0.0,
                 emb_dim_list=None, emb_type_list=None, attn_info=None, use_checkpoint=False,
                 image_shape=None, decode_fn_str_list=["pixel_shuffle"], img_dim=2, use_residual_conv=True):
        super().__init__()

        support_decode_fn_list = ["upsample", "pixel_shuffle", "conv_transpose"]
        assert img_dim in [1, 2, 3]
        
        for decode_fn_str in decode_fn_str_list:
            assert decode_fn_str in support_decode_fn_list

        conv_fn = get_conv_nd_fn(img_dim)
        self.use_checkpoint = use_checkpoint
        if image_shape is not None:
            image_shape = image_shape * 2
        conv_common_kwarg_dict = {
            "kernel_size": 3, "stride": 1, "padding": 1,
            "dropout_proba": dropout_proba,
            "norm": norm, "act": act,
            "emb_dim_list": emb_dim_list,
            "emb_type_list": emb_type_list,
            "attn_info": attn_info,
            "use_checkpoint": use_checkpoint,
            "image_shape": image_shape,
            "img_dim": img_dim
        }
        if isinstance(kernel_size, int):
            kernel_size = tuple(kernel_size for _ in range(img_dim))

        decode_middle_channel = 0
        decode_layer_list = []
        upsample_mode = None
        pixel_shuffle_fn = None
        conv_transpose_fn = None

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
            

        if "upsample" in decode_fn_str_list:
            upsample_layer = nn.Upsample(scale_factor=kernel_size, mode=upsample_mode)
            decode_layer_list.append(upsample_layer)
            decode_middle_channel += in_channels

        if "pixel_shuffle" in decode_fn_str_list:
            conv_before_pixel_shuffle = conv_fn(in_channels=in_channels,
                                        out_channels=in_channels *
                                        np.prod(kernel_size),
                                        kernel_size=1)
            pixel_shuffle = nn.Sequential(
                conv_before_pixel_shuffle,
                pixel_shuffle_fn(upscale_factor=kernel_size[0])
            )
            decode_layer_list.append(pixel_shuffle)
            decode_middle_channel += in_channels

        if "conv_transpose" in decode_fn_str_list:
            conv_transpose = conv_transpose_fn(in_channels, in_channels,
                                                    stride=kernel_size,
                                                    **conv_transpose_kwarg_dict)
            decode_layer_list.append(conv_transpose)
            decode_middle_channel += in_channels

        self.decode_layer_list = nn.ModuleList(decode_layer_list)
        if use_residual_conv:
            conv_block = ResNetBlockND
        else:
            conv_block = ConvBlockND
        self.concat_conv = conv_block(in_channels=decode_middle_channel,
                                      out_channels=out_channels, **conv_common_kwarg_dict)
        self.skip_conv = conv_block(in_channels=out_channels + skip_channels,
                                      out_channels=out_channels, **conv_common_kwarg_dict)
    def forward(self, x, *args):
        if self.use_checkpoint:
            return checkpoint(self._forward_impl, x, *args,
                              use_reentrant=False)
        else:
            return self._forward_impl(x, *args)
        
    def _forward_impl(self, x, skip, *args):

        decode_output_list = [decode_layer(x) for decode_layer in self.decode_layer_list]
        decode_output = torch.cat(decode_output_list, dim=1)
        decode_output = self.concat_conv(decode_output, *args)
        decode_output = torch.cat([decode_output, skip], dim=1)
        decode_output = self.skip_conv(decode_output)
        return decode_output
    
def get_upsample_mode_str(img_dim):
    mode_str_list = ["linear", "bilinear", "trilinear"]
    mode_str = mode_str_list[img_dim - 1]
    return mode_str
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, img_dim: int = 2):
        conv_fn = get_conv_nd_fn(img_dim)
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            conv_fn(in_channels, out_channels, kernel_size=1, bias=False),
            GroupNorm32(out_channels),
            nn.SiLU(),
        )
        self.mode_str = get_upsample_mode_str(img_dim)
        
    def forward(self, x, *args) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode=self.mode_str, align_corners=False)

class ASPPConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                dilation: int = 1, norm="batch", groups=1, act=DEFAULT_ACT, dropout_proba=0.0,
                emb_dim_list=[], emb_type_list=[], attn_info=None, use_checkpoint=False, image_shape=None,
                img_dim=2, separable=False, use_resnet_block=False):
        super().__init__()
        if use_resnet_block:
            conv_block = ResNetBlockND(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                       padding=dilation, dilation=dilation, norm=norm, groups=groups, act=act, bias=False,
                                       dropout_proba=dropout_proba, emb_dim_list=emb_dim_list, emb_type_list=emb_type_list, attn_info=attn_info,
                                       use_checkpoint=use_checkpoint, image_shape=image_shape, img_dim=img_dim, separable=separable)
        else:
            conv_block = ConvBlockND(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                       padding=dilation, dilation=dilation, norm=norm, groups=groups, act=act, bias=False,
                                       dropout_proba=dropout_proba, emb_dim_list=emb_dim_list, emb_type_list=emb_type_list, attn_info=attn_info,
                                       use_checkpoint=use_checkpoint, image_shape=image_shape, img_dim=img_dim, separable=separable)
        self.conv_block = conv_block
    def forward(self, x, *args):
        return self.conv_block(x, *args)
    
class ASPP(nn.Module):
    def __init__(
        self,
        in_channels: int, out_channels: int, atrous_rates: Iterable[int],
        norm="batch", groups=1, act=DEFAULT_ACT, dropout_proba=0.0,
        emb_dim_list=[], emb_type_list=[], attn_info=None, use_checkpoint=False, image_shape=None,
        img_dim=2, separable=False, use_resnet_block=False,
    ):
        super(ASPP, self).__init__()
        if use_resnet_block:
            conv_class = ResNetBlockND
        else:
            conv_class = ConvBlockND
        modules = [
            conv_class(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                        padding=0, dilation=1, norm=norm, groups=groups, act=act, bias=False,
                        dropout_proba=dropout_proba, emb_dim_list=emb_dim_list, emb_type_list=emb_type_list, attn_info=attn_info,
                        use_checkpoint=use_checkpoint, image_shape=image_shape, img_dim=img_dim, separable=separable)
        ]

        def get_aspp_conv_module(in_channels, out_channels, dilation):
            aspp_conv_kwargs = {
                "in_channels": in_channels, "out_channels": out_channels,
                "kernel_size":3, "padding": dilation, "dilation": dilation,
                "norm": norm, "groups": groups, "act":act, "dropout_proba":dropout_proba,
                "emb_dim_list": emb_dim_list, "emb_type_list":emb_type_list, "attn_info": attn_info,
                "use_checkpoint": use_checkpoint, "image_shape":image_shape, "img_dim":img_dim,
                "separable": separable
            }
            aspp_conv_module = conv_class(**aspp_conv_kwargs)
            return aspp_conv_module
        for rate in atrous_rates:
            modules.append(get_aspp_conv_module(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = ConvBlockND(len(self.convs) * out_channels, out_channels, kernel_size=1, padding=0, bias=False,
                                   norm=norm, act=act, dropout_proba=dropout_proba)

    def forward(self, x, *args) -> torch.Tensor:
        res = []
        for conv in self.convs:
            res.append(conv(x, *args))
        res = torch.cat(res, dim=1)
        return self.project(res)

class MultiDeepLabV3PlusDecoder(nn.Module):
    def __init__(self, encoder_channels, middle_channels=256, out_channels=2, atrous_rates=(12, 24, 36),
                 target_indices=[-3, -5], scale_factor=4, last_scale_factor=2,
                 norm="layer", act=DEFAULT_ACT, out_act="softmax", dropout_proba=0.0, separable=True,
                 emb_dim_list=[], emb_type_list=[], attn_info=None, use_checkpoint=False,
                 image_shape=None, img_dim=2, use_residual_conv=True):
        super().__init__()
        output_stride = 16
        self.low_res_feature_idx, self.high_res_feature_idx = target_indices

        self.mode_str = get_upsample_mode_str(img_dim)
        if use_residual_conv:
            conv_class = ResNetBlockND
        else:
            conv_class = ConvBlockND
        self.aspp = ASPP(
                encoder_channels[self.low_res_feature_idx],
                middle_channels,
                atrous_rates,
                norm=norm, groups=1, act=act, dropout_proba=dropout_proba,
                emb_dim_list=emb_dim_list, emb_type_list=emb_type_list, attn_info=attn_info,
                use_checkpoint=use_checkpoint, image_shape=image_shape, img_dim=img_dim,
                separable=separable, use_resnet_block=use_residual_conv
            )
        self.aspp_conv = ConvBlockND(middle_channels, middle_channels, kernel_size=3, padding=1,
                                     bias=False, norm=norm, act=act, separable=separable, use_checkpoint=use_checkpoint)
        self.up_1 = nn.Upsample(scale_factor=scale_factor, mode=self.mode_str)
        highres_in_channels = encoder_channels[self.high_res_feature_idx]
        highres_out_channels = 48  # proposed by authors of paper
        self.block1 = ConvBlockND(highres_in_channels, highres_out_channels, kernel_size=1, padding=0, bias=False, norm=norm, act=act)
        self.block2 = conv_class(highres_out_channels + middle_channels, middle_channels,
                                 kernel_size=3, padding=1, bias=False, norm=norm, act=act,
                                 separable=True, use_checkpoint=use_checkpoint)
        self.up_2 = nn.Upsample(scale_factor=last_scale_factor, mode=self.mode_str)
        self.output_block = OutputND(middle_channels, out_channels, act=out_act, img_dim=img_dim)

    def forward(self, features, *args):
        aspp_features = self.aspp(features[self.low_res_feature_idx], *args)
        aspp_features = self.aspp_conv(aspp_features)
        aspp_features = self.up_1(aspp_features)
        high_res_features = self.block1(features[self.high_res_feature_idx])
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features = self.block2(concat_features)
        fused_features = self.up_2(fused_features)
        output = self.output_block(fused_features)
        return output
    
class OutputND(nn.Module):
    def __init__(self, in_channels, out_channels, act=None, img_dim=2):
        super().__init__()
        conv_fn = get_conv_nd_fn(img_dim)
        conv_out_channels = in_channels
        conv_common_kwarg_dict = {
            "kernel_size": 3, "stride": 1, "padding": 1,
            "norm": None, "act": act, "bias": False, "dropout_proba": 0.0,
            "emb_dim_list": [],
            "emb_type_list": [],
            "attn_info": None,
            "use_checkpoint": False,
            "img_dim": img_dim
        }
        self.conv_5x5 = conv_fn(in_channels=in_channels,
                                  out_channels=conv_out_channels,
                                  kernel_size=5, padding=2)
        self.conv_3x3 = conv_fn(in_channels=in_channels,
                                  out_channels=conv_out_channels,
                                  kernel_size=3, padding=1)
        self.concat_conv = ConvBlockND(in_channels=conv_out_channels * 2,
                                       out_channels=out_channels,
                                      **conv_common_kwarg_dict)

    def forward(self, x):
        conv_5x5 = self.conv_5x5(x)
        conv_3x3 = self.conv_3x3(x)
        output = torch.cat([conv_5x5, conv_3x3], dim=1)
        output = self.concat_conv(output)
        return output

class MaxPool1d(nn.MaxPool1d):
    def forward(self, x, *args):
        return super().forward(x)

class MaxPool2d(nn.MaxPool2d):
    def forward(self, x, *args):
        return super().forward(x)

class MaxPool3d(nn.MaxPool3d):
    def forward(self, x, *args):
        return super().forward(x)

class AvgPool1d(nn.AvgPool1d):
    def forward(self, x, *args):
        return super().forward(x)

class AvgPool2d(nn.AvgPool2d):
    def forward(self, x, *args):
        return super().forward(x)

class AvgPool3d(nn.AvgPool3d):
    def forward(self, x, *args):
        return super().forward(x)

def get_maxpool_nd(img_dim):
    assert img_dim in [1, 2, 3]
    maxpool_module = None
    if img_dim == 1:
        maxpool_module = MaxPool1d
    elif img_dim == 2:
        maxpool_module = MaxPool2d
    elif img_dim == 3:
        maxpool_module = MaxPool3d
    return maxpool_module

def get_avgpool_nd(img_dim):
    assert img_dim in [1, 2, 3]
    avgpool_module = None
    if img_dim == 1:
        avgpool_module = AvgPool1d
    elif img_dim == 2:
        avgpool_module = AvgPool2d
    elif img_dim == 3:
        avgpool_module = AvgPool3d
    return avgpool_module


class MultiInputSequential(nn.Sequential):
    def forward(self, x, *args):
        for module in self:
            x = module(x, *args)
        return x
    
class ClassificationHeadSimple(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_proba, act, img_dim=2):
        super(ClassificationHeadSimple, self).__init__()
        # Global Average Pooling Layer
        if img_dim == 1:
            self.gap_layer = nn.AdaptiveAvgPool1d((1))
        elif img_dim == 2:
            self.gap_layer = nn.AdaptiveAvgPool2d((1, 1))
        elif img_dim == 3:
            self.gap_layer = nn.AdaptiveAvgPool3d((1, 1, 1))

        # First fully connected layer
        self.fc_1 = nn.Linear(in_channels, in_channels * 2)
        self.drop_1 = nn.Dropout(p=dropout_proba, inplace=INPLACE)
        self.act_1 = nn.ReLU6(inplace=INPLACE)

        # Second fully connected layer
        self.fc_2 = nn.Linear(in_channels * 2, in_channels)
        self.drop_2 = nn.Dropout(p=dropout_proba, inplace=INPLACE)
        self.act_2 = nn.ReLU6(inplace=INPLACE)

        # Dropout layer

        # Third fully connected layer
        self.fc_3 = nn.Linear(in_channels, in_channels // 2)
        self.drop_3 = nn.Dropout(p=dropout_proba, inplace=INPLACE)
        self.act_3 = nn.ReLU6(inplace=INPLACE)
        # Output layer
        self.fc_out = nn.Linear(in_channels // 2, num_classes)
        self.last_act = get_act(act)

    def forward(self, x):
        x = self.gap_layer(x)
        x = x.flatten(start_dim=1, end_dim=-1)

        x = self.fc_1(x)
        x = self.drop_1(x)
        x = self.act_1(x)
        x = self.fc_2(x)
        x = self.drop_2(x)
        x = self.act_2(x)

        x = self.fc_3(x)
        x = self.drop_3(x)
        x = self.act_3(x)
        x = self.fc_out(x)
        x = self.last_act(x)

        return x
    

class ClassificationHeadDeepLab(nn.Module):
    def __init__(self, in_channels, encoder_channels, num_classes, 
                 norm, act, dropout_proba, class_act,
                 image_shape=None, use_checkpoint=False, img_dim=2):
        super(ClassificationHeadSimple, self).__init__()
        # Global Average Pooling Layer
        avg_pool_class = None
        self.gap_layer = None
        if img_dim == 1:
            avg_pool_class = nn.AvgPool1d
            self.gap_layer = nn.AdaptiveAvgPool1d((1))
        elif img_dim == 2:
            avg_pool_class = nn.AvgPool2d
            self.gap_layer = nn.AdaptiveAvgPool2d((1, 1))
        elif img_dim == 3:
            avg_pool_class = nn.AvgPool3d
            self.gap_layer = nn.AdaptiveAvgPool3d((1, 1, 1))
        

        middle_channels = 256
        target_indices = [-4, -3, -2]
        pool_size_1, pool_size_2, pool_size_3 = (8, 4, 2)
        # low feature size 64 expected, 
        self.low_res_feature_idx, self.middle_res_feature_idx, self.high_res_feature_idx = target_indices
        atrous_rates_1 = (6, 12, 18, 24)
        atrous_rates_2 = (6, 12, 18, 24)
        
        self.aspp_1 = ASPP(
                encoder_channels[self.low_res_feature_idx],
                middle_channels,
                atrous_rates_1,
                norm=norm, groups=1, act=act, dropout_proba=dropout_proba,
                emb_dim_list=[], emb_type_list=[], attn_info=None,
                use_checkpoint=use_checkpoint, image_shape=image_shape, img_dim=img_dim,
                separable=True, use_resnet_block=False
        )
        self.avg_pool_1 = avg_pool_class(kernel_size=pool_size_1, stride=pool_size_1)
        self.aspp_2 = ASPP(
                encoder_channels[self.middle_res_feature_idx],
                middle_channels,
                atrous_rates_2,
                norm=norm, groups=1, act=act, dropout_proba=dropout_proba,
                emb_dim_list=[], emb_type_list=[], attn_info=None,
                use_checkpoint=use_checkpoint, image_shape=image_shape, img_dim=img_dim,
                separable=True, use_resnet_block=False
        )
        self.avg_pool_2 = avg_pool_class(kernel_size=pool_size_2, stride=pool_size_2)
        self.aspp_conv = ConvBlockND(middle_channels, middle_channels, kernel_size=3, padding=1,
                                     bias=False, norm=norm, act=act, separable=True, use_checkpoint=use_checkpoint)
        # First fully connected layer
        self.fc_1 = nn.Linear(in_channels, in_channels * 2)
        self.drop_1 = nn.Dropout(p=dropout_proba, inplace=INPLACE)
        self.act_1 = nn.ReLU6(inplace=INPLACE)

        # Second fully connected layer
        self.fc_2 = nn.Linear(in_channels * 2, in_channels)
        self.drop_2 = nn.Dropout(p=dropout_proba, inplace=INPLACE)
        self.act_2 = nn.ReLU6(inplace=INPLACE)

        # Third fully connected layer
        self.fc_3 = nn.Linear(in_channels, in_channels * 2)
        self.drop_3 = nn.Dropout(p=dropout_proba, inplace=INPLACE)
        self.act_3 = nn.ReLU6(inplace=INPLACE)

        # Third fully connected layer
        self.fc_4 = nn.Linear(in_channels * 2, in_channels)
        self.drop_4 = nn.Dropout(p=dropout_proba, inplace=INPLACE)
        self.act_4 = nn.ReLU6(inplace=INPLACE)

        # Output layer
        self.fc_out = nn.Linear(in_channels, num_classes)
        self.last_act = get_act(act)

    def forward(self, x):
        x = self.gap_layer(x)
        x = x.flatten(start_dim=1, end_dim=-1)

        x = self.fc_1(x)
        x = self.drop_1(x)
        x = self.act_1(x)
        x = self.fc_2(x)
        x = self.drop_2(x)
        x = self.act_2(x)

        x = self.fc_3(x)
        x = self.drop_3(x)
        x = self.act_3(x)
        x = self.fc_4(x)
        x = self.drop_4(x)
        x = self.act_4(x)

        x = self.fc_out(x)
        x = self.last_act(x)

        return x    
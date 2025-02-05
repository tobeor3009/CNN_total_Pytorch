from collections import namedtuple
from packaging import version
from functools import wraps
from functools import partial
import math
import numpy as np
from typing import NamedTuple, Optional, Any
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

@dataclass
class Return:
    pred: Optional[torch.Tensor] = None
    seg_output: Optional[torch.Tensor] = None
    class_output: Optional[torch.Tensor] = None
    recon_output: Optional[torch.Tensor] = None
    validity_output: Optional[torch.Tensor] = None

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
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor):

        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        x = x.permute(0, 2, 1)
        return F.rms_norm(x, (self.dim,), self.weight, self.eps).permute(0, 2, 1)
    
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
        norm_layer="group"
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
            self.norm = RMSNorm(channels)
        elif norm_layer== "group":
            self.norm = GroupNorm32(channels)
        
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
        x = self.norm(x)
        qkv = self.qkv(x)
        h = self.attention(qkv)
        h = self.proj_out(h)
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
        if emb is None:
            emb = emb_block(emb)
            emb_shape = emb.shape
            emb = emb.view(*emb_shape, *unsqueeze_dim)
            if emb_type == "cond":
                scale = emb
                shift = torch.zeros_like(emb)
            else:
                scale, shift = emb.chunk(2, dim=1)
            scale_shift_list.append([scale, shift])
        else:
            scale_shift_list.append([0, 0])
    return scale_shift_list

def apply_embedding(x, scale_shift_list, scale_bias=1):
    for scale, shift in scale_shift_list:
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
                 stride=1, padding='same',
                 norm=nn.LayerNorm, groups=1, act=DEFAULT_ACT, bias=False, dropout_proba=0.0, image_shape=None, img_dim=2):
        super().__init__()
        conv_fn = get_conv_nd_fn(img_dim)
        self.conv = conv_fn(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups, bias=bias)
        
        if (not bias) and (norm is not None):
            if image_shape is None:
                image_shape = in_channels
            else:
                image_shape = (in_channels, *image_shape)
            self.norm_layer = norm(image_shape)
        else:
            self.norm_layer = nn.Identity()
        self.act_layer = get_act(act)
        self.dropout_layer = nn.Dropout(p=dropout_proba, inplace=INPLACE)
    
    def forward(self, x, scale_shift_list=[]):
        norm = self.norm_layer(x)
        drop = self.dropout_layer(norm)
        act = self.act_layer(drop)
        conv = self.conv(act)
        conv = apply_embedding(conv, scale_shift_list)
        return conv

#attn_info.keys = ["emb_type_list", "num_heads", "full_attn"]
class ResNetBlockND(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding='same',
                 norm="batch", groups=1, act=DEFAULT_ACT, bias=False, dropout_proba=0.0,
                 emb_dim_list=[], emb_type_list=[], attn_info=None, use_checkpoint=False, image_shape=None, img_dim=2):
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
                emb_block = BaseBlockND(emb_dim, out_channels * 2, kernel_size,
                                        1, padding, norm, groups, act, bias, dropout_proba=0.0, img_dim=img_dim)
            else:
                raise NotImplementedError(f"emb_type must be in [seq, cond, nd]")
            emb_block_list.append(emb_block)
        self.emb_block_list = nn.ModuleList(emb_block_list)
        self.emb_type_list = emb_type_list

        self.block_1 = BaseBlockND(in_channels, out_channels, kernel_size,
                                    1, padding, norm, groups, act, bias, dropout_proba=dropout_proba,
                                    image_shape=image_shape, img_dim=img_dim)
        self.block_2 = BaseBlockND(out_channels, out_channels, kernel_size,
                                    stride, padding, norm, groups, act, bias, dropout_proba=dropout_proba, 
                                    image_shape=image_shape, img_dim=img_dim)

        if in_channels != out_channels or stride != 1:
            self.residiual_conv = conv_fn(in_channels, out_channels, 1, stride, bias=False)
        else:
            self.residiual_conv = nn.Identity()

        if attn_info is None:
            self.attn = nn.Identity()
        elif attn_info["full_attn"] is None:
            self.attn = nn.Identity()
        elif attn_info["full_attn"] is True:
            self.attn = AttentionBlock(out_channels, attn_info["num_heads"], use_checkpoint=use_checkpoint)
        else:
            self.attn = LinearAttention(out_channels, attn_info["num_heads"], attn_info["dim_head"], use_checkpoint=use_checkpoint)


    def forward(self, x, *args):
        if self.use_checkpoint:
            conv_output = checkpoint(self._forward_conv, x, *args,
                              use_reentrant=False)
        else:
            conv_output = self._forward_conv(x, *args)
        attn_output = self._forward_attn(conv_output)
        return attn_output
    
    def _forward_conv(self, x, *emb_args):
        skip_x = x
        scale_shift_list = get_scale_shift_list(self.emb_block_list, self.emb_type_list,
                                                emb_args, self.img_dim)
        x = self.block_1(x)
        x = self.block_2(x, scale_shift_list)
        x = x + self.residiual_conv(skip_x)
        return x

    def _forward_attn(self, x):
        x = self.attn(x)
        return x

class ResNetBlockNDSkip(ResNetBlockND):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, skip, *args):
        if self.use_checkpoint:
            return checkpoint(self._forward_impl, x, skip, *args,
                              use_reentrant=False)
        else:
            return self._forward_impl(x, skip, *args)

    def _forward_impl(self, x, skip, *emb_args):
        x = torch.cat([x, skip], dim=1)
        x = super().forward(x, *emb_args)
        return x

class ConvBlockND(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding='same',
                 norm="batch", groups=1, act=DEFAULT_ACT, bias=False, dropout_proba=0.0,
                 emb_dim_list=[], emb_type_list=[], attn_info=None, use_checkpoint=False,
                 image_shape=None, img_dim=2):
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
                                        1, padding, norm, groups, act, bias, dropout_proba=0.0, img_dim=img_dim)
            else:
                raise NotImplementedError(f"emb_type must be in [seq, cond, nd]")
            emb_block_list.append(emb_block)
        self.emb_block_list = nn.ModuleList(emb_block_list)
        self.emb_type_list = emb_type_list

        self.block_1 = BaseBlockND(in_channels, out_channels, kernel_size,
                                    stride, padding, norm, groups, act, bias,
                                    dropout_proba=dropout_proba, image_shape=image_shape, img_dim=img_dim)
        if attn_info is None:
            self.attn = nn.Identity()
        elif attn_info["full_attn"] is None:
            self.attn = nn.Identity()
        elif attn_info["full_attn"] is True:
            self.attn = AttentionBlock(out_channels, attn_info["num_heads"])
        else:
            self.attn = LinearAttention(out_channels, attn_info["num_heads"], attn_info["dim_head"])

    def forward(self, x, *args):
        if self.use_checkpoint:
            return checkpoint(self._forward_impl, x, *args,
                              use_reentrant=False)
        else:
            return self._forward_impl(x, *args)

    def _forward_impl(self, x, *emb_args):
        scale_shift_list = get_scale_shift_list(self.emb_block_list, self.emb_type_list,
                                                emb_args, self.img_dim)
        x = self.block_1(x, scale_shift_list)
        x = self.attn(x)
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
                 emb_dim_list=None, emb_type_list=None, attn_info=None, use_checkpoint=False,
                 image_shape=None, decode_fn_str_list=["pixel_shuffle"], img_dim=2):
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
        self.concat_conv = ResNetBlockND(in_channels=decode_middle_channel,
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
        self.concat_conv = ResNetBlockND(in_channels=conv_out_channels * 2,
                                       out_channels=out_channels,
                                      **conv_common_kwarg_dict)
        self.act = get_act(act)

    def forward(self, x):
        conv_5x5 = self.conv_5x5(x)
        conv_3x3 = self.conv_3x3(x)
        output = torch.cat([conv_5x5, conv_3x3], dim=1)
        output = self.concat_conv(output)
        output = self.act(output)
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
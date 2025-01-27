
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from collections import namedtuple

from torch.utils.checkpoint import checkpoint
from timm.models.layers import DropPath, to_2tuple
import numpy as np
from .diffusion_layer import AttentionBlock, GroupNorm32
from .diffusion_layer import apply_embedding
from src.model.inception_resnet_v2.common_module.layers import get_act, get_norm, DEFAULT_ACT, INPLACE
from src.model.swin_transformer.model_2d.swin_layers import window_partition, window_reverse, check_hasattr_and_init
from src.model.swin_transformer.model_2d.swin_layers import Mlp, WindowAttention
from src.model.swin_transformer.model_2d.swin_layers import DROPOUT_INPLACE, InstanceNormChannelLast
from .diffusion_layer import get_scale_shift_list, apply_embedding
from einops import rearrange, repeat
from functools import partial
from functools import wraps
from packaging import version


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
        print(x.shape)
        print(self.lambd(x).shape)
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
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)

class GroupNorm32(nn.GroupNorm):
    def __init__(self, num_channels: int, *args, **kwargs) -> None:
        num_groups = min(32, num_channels)
        super().__init__(num_groups, num_channels, *args, **kwargs)

def get_norm_layer(dim, norm_layer_str):

    if norm_layer_str == "rms":
        norm_layer = RMSNorm(dim)
    elif norm_layer_str == "group":
        norm_layer = GroupNorm32(dim)
    elif norm_layer_str is None:
        norm_layer = nn.Identity()
    else:
        raise NotImplementedError()
    return norm_layer
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
                 norm_layer="rms", act_layer=DEFAULT_ACT, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = get_norm_layer(dim, norm_layer)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size),
                                    num_heads=num_heads, qkv_bias=qkv_bias,
                                    qkv_drop=qkv_drop, attn_drop=attn_drop, proj_drop=drop,
                                    pretrained_window_size=to_2tuple(pretrained_window_size),
                                    cbp_dim=np.clip(np.prod(input_resolution) // 16, 256, 1024))

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = get_norm_layer(dim, norm_layer)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1,
                                             self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, scale_shift_list=[]):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size, {L} != {H}, {W}"

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x,
                                   shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        # nW*B, window_size, window_size, C
        x_windows = window_partition(shifted_x, self.window_size)
        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        # nW*B, window_size*window_size, C
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1,
                                         self.window_size,
                                         self.window_size, C)
        shifted_x = window_reverse(attn_windows,
                                   self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x,
                           shifts=(self.shift_size,
                                   self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        norm = self.norm1(x)
        norm = apply_embedding(norm, scale_shift_list)
        x = shortcut + self.drop_path(norm)
        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops
    
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = np.array([img_size[0] // patch_size[0],
                                       img_size[1] // patch_size[1]])
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = get_norm_layer(embed_dim, norm_layer)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * \
            (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

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

    def __init__(self, input_resolution, dim, norm_layer="rms"):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = get_norm_layer(2 * dim, norm_layer)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size {L} != {H}, {W}"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops

class PatchExpanding(nn.Module):
    def __init__(self, input_resolution, dim,
                 return_vector=True, dim_scale=2, norm_layer="rms"):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.return_vector = return_vector
        self.dim_scale = dim_scale
        
        pixel_conv = nn.Conv2d(dim, dim * (dim_scale ** 2) // 2,
                               kernel_size=1, padding=0, bias=False)
        if dim_scale == 1:
            self.pixel_shuffle = pixel_conv
        else:
            self.pixel_shuffle = nn.Sequential(
                            pixel_conv,
                            nn.PixelShuffle(dim_scale)
            )
        self.norm_layer = get_norm_layer(dim // 2, norm_layer)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size {L} != {H}, {W}"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.pixel_shuffle(x)
        
        x = x.permute(0, 2, 3, 1).view(B, -1, self.dim // 2)
        x = self.norm_layer(x)
        if not self.return_vector:
            x = x.permute(0, 2, 1).view(B, self.dim // 2,
                                        H * self.dim_scale,
                                        W * self.dim_scale)
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
                 drop_path=0., norm_layer="rms", act_layer=DEFAULT_ACT, downsample=None, upsample=None,
                 use_checkpoint=False, pretrained_window_size=0, emb_dim_list=[], emb_type_list=[], img_dim=2):

        super().__init__()
        assert len(emb_dim_list) == len(emb_type_list)
        self.dim = dim
        self.input_resolution = input_resolution

        self.depth = depth
        self.use_checkpoint = use_checkpoint
        emb_block_list = []
        for emb_dim in emb_dim_list:
            emb_block = nn.Sequential(
                                        act_layer,
                                        nn.Linear(emb_dim, dim * 2)
                                    )
            emb_block_list.append(emb_block)
        self.emb_block_list = nn.ModuleList(emb_block_list)
        self.emb_type_list = emb_type_list

        # build blocks
        self.blocks = nn.ModuleList([
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
            for i in range(depth)])
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution,
                                         dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = nn.Identity()
        if upsample is not None:
            self.upsample = upsample(input_resolution,
                                     dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = nn.Identity()

    def forward(self, x, *emb_args):
        scale_shift_list = get_scale_shift_list(self.emb_block_list, self.emb_type_list, emb_args)
        for blk in self.blocks:
            x = process_checkpoint_block(self.use_checkpoint, blk, x, scale_shift_list)
        x = process_checkpoint_block(self.use_checkpoint, self.downsample, x)
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
                 drop_path=0., norm_layer=nn.LayerNorm, act_layer=DEFAULT_ACT, downsample=None, upsample=None,
                 use_checkpoint=False, pretrained_window_size=0, emb_dim_list=[], emb_type_list=[], img_dim=2):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution,
                                         dim=dim, norm_layer=norm_layer)
            dim *= 2
            num_heads *= 2
            window_size = max(window_size // 2, 2)
            input_resolution = [input_resolution[0] // 2,
                                input_resolution[1] // 2]
        else:
            self.downsample = nn.Identity()
        if upsample is not None:
            self.upsample = upsample(input_resolution,
                                     dim=dim, norm_layer=norm_layer)
            dim //= 2
            num_heads = max(num_heads // 2, 1)
            window_size *= 2
            input_resolution = [input_resolution[0] * 2,
                                input_resolution[1] * 2]
        else:
            self.upsample = nn.Identity()

        emb_block_list = []
        for emb_dim in emb_dim_list:
            emb_block = nn.Sequential(
                                        act_layer,
                                        nn.Linear(emb_dim, dim * 2)
                                    )
            emb_block_list.append(emb_block)
        self.emb_block_list = nn.ModuleList(emb_block_list)
        
        # build blocks
        self.blocks = nn.ModuleList([
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
            for i in range(depth)])

    def forward(self, x, *args):
        scale_shift_list = []
        for emb_block, emb in zip(self.emb_block_list, args):
            emb = emb_block(emb)
            if emb.ndim == 2:
                emb = emb.unsqueeze(1)
            scale_shift = emb.chunk(2, dim=2)
            scale_shift_list.append(scale_shift)
        if self.downsample is not None:
            if self.use_checkpoint:
                x = checkpoint(self.downsample, x,
                               use_reentrant=False)
            else:
                x = self.downsample(x)
        if self.upsample is not None:
            if self.use_checkpoint:
                x = checkpoint(self.upsample, x,
                               use_reentrant=False)
            else:
                x = self.upsample(x)
        x = self.process_block(x, scale_shift_list)
        return x
    def process_block(self, x, scale_shift_list):
        if self.use_residual:
            shortcut = x
            if self.use_checkpoint:
                x = checkpoint(self.blocks[0], x, scale_shift_list,
                            use_reentrant=False)
                x = checkpoint(self.blocks[1], x,
                            use_reentrant=False)
            else:
                x = self.blocks[0](x, scale_shift_list)
                x = self.blocks[1](x, scale_shift_list)
            x = x + shortcut
        else:
            for blk in self.blocks:
                if self.use_checkpoint:
                    x = checkpoint(blk, x, scale_shift_list,
                                use_reentrant=False)
                else:
                    x = blk(x, scale_shift_list)
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

class BasicDecodeLayer(nn.Module):
    def __init__(self, dim, skip_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., qkv_drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, act_layer=DEFAULT_ACT, upsample=None,
                 use_checkpoint=False, pretrained_window_size=0,
                 emb_dim_list=[], use_residual=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.use_residual = use_residual
        
        if use_residual:
            assert depth == 2, "residual depth must be 2"
        # patch merging layer
        assert upsample is not None
        self.upsample = upsample(input_resolution,
                                    dim=dim, norm_layer=norm_layer)
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
            for i in range(depth)])
        dim //= 2
        num_heads = max(num_heads // 2, 1)
        window_size *= 2
        input_resolution = [input_resolution[0] * 2,
                            input_resolution[1] * 2]

        emb_block_list = []
        for emb_dim in emb_dim_list:
            emb_block = nn.Sequential(
                                        act_layer,
                                        nn.Linear(emb_dim, dim * 2)
                                    )
            emb_block_list.append(emb_block)
        self.emb_block_list = nn.ModuleList(emb_block_list)
        
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
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])
        
    def forward(self, x, skip, *args):
        scale_shift_list = []
        for emb_block, emb in zip(self.emb_block_list, args):
            emb = emb_block(emb)
            if emb.ndim == 2:
                emb = emb.unsqueeze(1)
            scale_shift = emb.chunk(2, dim=2)
            scale_shift_list.append(scale_shift)
        x = self.process_block(self.blocks_before_skip, x, scale_shift_list)
        if self.use_checkpoint:
            x = checkpoint(self.upsample, x,
                            use_reentrant=False)
        else:
            x = self.upsample(x)
        x = torch.cat([x, skip], dim=-1)
        x = self.mlp_after_skip(x)
        x = self.process_block(self.blocks_after_skip, x, scale_shift_list)
        return x
    
    def process_block(self, blk_list, x, scale_shift_list):
        if self.use_residual:
            shortcut = x
            if self.use_checkpoint:
                x = checkpoint(blk_list[0], x, scale_shift_list,
                            use_reentrant=False)
                x = checkpoint(blk_list[1], x,
                            use_reentrant=False)
            else:
                x = blk_list[0](x, scale_shift_list)
                x = blk_list[1](x, scale_shift_list)
            x = x + shortcut
        else:
            for blk in blk_list:
                if self.use_checkpoint:
                    x = checkpoint(blk, x, scale_shift_list,
                                use_reentrant=False)
                else:
                    x = blk(x, scale_shift_list)
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

class BaseBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding='same',
                 norm=nn.GroupNorm, groups=1, act=DEFAULT_ACT, bias=False):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups, bias=bias)
        if not bias:
            self.norm_layer = norm(out_channels)
        else:
            self.norm_layer = nn.Identity()
        self.act_layer = get_act(act)

    def forward(self, x, scale_shift_list=None):
        conv = self.conv(x)
        norm = self.norm_layer(conv)
        if exists(scale_shift_list):
            for scale_shift in scale_shift_list:
                scale, shift = scale_shift
                norm = norm * (scale + 1) + shift
        act = self.act_layer(norm)
        return act

class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding='same',
                 norm=nn.GroupNorm, groups=1, act=DEFAULT_ACT, bias=False,
                 emb_dim_list=[], emb_type_list=[], use_checkpoint=False):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        # you always have time embedding
        emb_block_list = []
        for emb_dim, emb_type in zip(emb_dim_list, emb_type_list):
            if emb_type == "seq":
                emb_block = nn.Sequential(
                                            nn.SiLU(),
                                            nn.Linear(emb_dim, out_channels * 2)
                                        )
            elif emb_type == "2d":
                emb_block = BaseBlock2D(emb_dim, out_channels * 2, kernel_size,
                                        1, padding, norm, groups, act, bias)
            else:
                raise Exception("emb_type must be seq or 2d")
            emb_block_list.append(emb_block)
        self.emb_block_list = nn.ModuleList(emb_block_list)

        self.block_1 = BaseBlock2D(in_channels, out_channels, kernel_size,
                                    stride, padding, norm, groups, act, bias)
        
    def forward(self, x, *args):
        if self.use_checkpoint:
            return checkpoint(self._forward_impl, x, *args,
                              use_reentrant=False)
        else:
            return self._forward_impl(x, *args)

    def _forward_impl(self, x, *args):
        scale_shift_list = []
        for emb_block, emb in zip(self.emb_block_list, args):
            emb = emb_block(emb)
            emb = rearrange(emb, 'b c -> b c 1 1')
            scale_shift = emb.chunk(2, dim=1)
            scale_shift_list.append(scale_shift)
        x = self.block_1(x, scale_shift_list)
        return x
        
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



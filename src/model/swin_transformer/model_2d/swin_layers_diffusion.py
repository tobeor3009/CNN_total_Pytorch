
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from timm.models.layers import DropPath, to_2tuple
import numpy as np
from ..layers import get_act
from .swin_layers import DEFAULT_ACT, DROPOUT_INPLACE
from .swin_layers import window_partition, window_reverse
from .swin_layers import Mlp, WindowAttention
from einops import rearrange

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
class SkipConv1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.skip_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(out_channels)
    def forward(self, *args):
        # expected shape: [B, N, C]
        x = torch.cat(args, dim=2)
        x = self.skip_conv(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.norm(x) 
        return x
    
class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dim_head=None):
        super().__init__()

        self.num_heads = num_heads
        if dim_head is None:
            dim_head = dim
        self.dim_head = dim_head
        self.scale = (dim_head / num_heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim_head * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim_head, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(b, n, 3, self.num_heads, self.dim_head // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  
        
        k = k.softmax(dim=-2)
        q = q * self.scale

        context = torch.einsum('bhid,bhjd->bhij', k, v)
        out = torch.einsum('bhij,bhid->bhjd', context, q)

        out = out.reshape(b, n, c)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, dim_head=None):
        super().__init__()
        self.num_heads = num_heads
        if dim_head is None:
            dim_head = dim
        self.dim_head = dim_head
        self.scale = (dim_head / num_heads) ** -0.5

        self.to_qkv = nn.Linear(dim, dim_head * 3, bias = False)
        self.to_out = nn.Linear(dim_head, dim)

    def forward(self, x):
        b, n, c = x.shape

        qkv = self.to_qkv(x).reshape(b, n, 3, self.num_heads, self.dim_head // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  

        attn = (q @ k.transpose(-2, -1) ) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().reshape(b, n, c)

        return self.to_out(out)
    
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
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=DEFAULT_ACT, norm_layer=nn.LayerNorm, pretrained_window_size=0, use_residual=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_residual = use_residual
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(num_channels=dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size),
                                    num_heads=num_heads, qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop,
                                    pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(num_channels=dim)
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

    def forward(self, x, scale_shift_list=None):
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
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        if exists(scale_shift_list):
            for scale_shift in scale_shift_list:
                scale, shift = scale_shift
                x = x * (scale + 1) + shift
        if self.use_residual:
            return shortcut + x
        else:
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

class CondBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=DEFAULT_ACT, norm_layer=nn.LayerNorm, pretrained_window_size=0):
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

        self.norm1 = norm_layer(num_channels=dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size),
                                    num_heads=num_heads, qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop,
                                    pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(num_channels=dim)
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

    def forward(self, x, scale_shift_list=None):
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
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        if exists(scale_shift_list):
            for scale_shift in scale_shift_list:
                scale, shift = scale_shift
                x = x * (scale + 1) + shift
        return shortcut + x

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
        patches_resolution = [img_size[0] // patch_size[0],
                              img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(num_channels=embed_dim)
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
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(num_channels=2 * dim)

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
                 return_vector=True, dim_scale=2,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.return_vector = return_vector
        self.dim_scale = dim_scale
        self.pixel_shuffle_conv_1 = nn.Conv2d(dim, dim * (dim_scale ** 2) // 2,
                                              kernel_size=1, padding=0, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(dim_scale)
        self.norm_layer = norm_layer(num_channels=dim // 2)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size {L} != {H}, {W}"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.pixel_shuffle_conv_1(x)
        x = self.pixel_shuffle(x)
        x = x.permute(0, 2, 3, 1).view(B, -1, self.dim // 2)
        x = self.norm_layer(x)
        if not self.return_vector:
            x = x.permute(0, 2, 1).view(B, self.dim // 2,
                                        H * self.dim_scale,
                                        W * self.dim_scale)
        return x
class AttnBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=DEFAULT_ACT, norm_layer=nn.LayerNorm, pretrained_window_size=0, full_attn=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(num_channels=dim)
        attn_layer = Attention if full_attn else LinearAttention 
        self.attn = attn_layer(dim, num_heads=num_heads)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(num_channels=dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)


    def forward(self, x, scale_shift_list=None):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size, {L} != {H}, {W}"

        shortcut = x
        x = self.attn(x)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        if exists(scale_shift_list):
            for scale_shift in scale_shift_list:
                scale, shift = scale_shift
                x = x * (scale + 1) + shift
            
        return shortcut + x

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
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, upsample=None,
                 use_checkpoint=False, pretrained_window_size=0, time_emb_dim=None, class_emb_dim=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim * 2)
        ) if exists(time_emb_dim) else None
        self.class_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(class_emb_dim, dim * 2)
        ) if exists(class_emb_dim) else None
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (
                                     i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(
                                     drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

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
            self.downsample = None
        if upsample is not None:
            self.upsample = upsample(input_resolution,
                                     dim=dim, norm_layer=norm_layer)
            dim //= 2
            num_heads = max(num_heads // 2, 1)
            window_size *= 2
            input_resolution = [input_resolution[0] * 2,
                                input_resolution[1] * 2]
        else:
            self.upsample = None
        
    def forward(self, x, time_emb=None, class_emb=None):
        scale_shift_list = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b 1 c')
            time_scale_shift = time_emb.chunk(2, dim = 2)
        if exists(self.class_mlp) and exists(class_emb):
            class_emb = self.class_mlp(class_emb)
            class_emb = rearrange(class_emb, 'b c -> b 1 c')
            class_scale_shift = class_emb.chunk(2, dim = 2)
        scale_shift_list = [time_scale_shift, class_scale_shift]
        for idx, blk in enumerate(self.blocks):
            if idx == 0:
                blk_scale_shift = scale_shift_list
            else:
                blk_scale_shift = None
            if self.use_checkpoint:
                x = checkpoint(blk, x, blk_scale_shift,
                               use_reentrant=False)
            else:
                x = blk(x, blk_scale_shift)
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
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class BasicLayerV2(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, upsample=None,
                 use_checkpoint=False, pretrained_window_size=0,
                 time_emb_dim=None, class_emb_dim=None, use_residual=True):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
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
            self.downsample = None
        if upsample is not None:
            self.upsample = upsample(input_resolution,
                                     dim=dim, norm_layer=norm_layer)
            dim //= 2
            num_heads = max(num_heads // 2, 1)
            window_size *= 2
            input_resolution = [input_resolution[0] * 2,
                                input_resolution[1] * 2]
        else:
            self.upsample = None

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim * 2)
        ) if exists(time_emb_dim) else None
        self.class_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(class_emb_dim, dim * 2)
        ) if exists(class_emb_dim) else None

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (
                                     i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(
                                     drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size,
                                 use_residual=use_residual)
            for i in range(depth)])
        
    def forward(self, x, time_emb=None, class_emb=None):
        scale_shift_list = []
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b 1 c')
            time_scale_shift = time_emb.chunk(2, dim = 2)
            scale_shift_list.append(time_scale_shift)
        if exists(self.class_mlp) and exists(class_emb):
            class_emb = self.class_mlp(class_emb)
            class_emb = rearrange(class_emb, 'b c -> b 1 c')
            class_scale_shift = class_emb.chunk(2, dim = 2)
            scale_shift_list.append(class_scale_shift)

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
        for idx, blk in enumerate(self.blocks):
            if idx == 0:
                blk_scale_shift = scale_shift_list
            else:
                blk_scale_shift = None
            if self.use_checkpoint:
                x = checkpoint(blk, x, blk_scale_shift,
                               use_reentrant=False)
            else:
                x = blk(x, blk_scale_shift)
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
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class CondLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, upsample=None,
                 use_checkpoint=False, pretrained_window_size=0, time_emb_dim=None, class_emb_dim=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
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
            self.downsample = None
        if upsample is not None:
            self.upsample = upsample(input_resolution,
                                     dim=dim, norm_layer=norm_layer)
            dim //= 2
            num_heads = max(num_heads // 2, 1)
            window_size *= 2
            input_resolution = [input_resolution[0] * 2,
                                input_resolution[1] * 2]
        else:
            self.upsample = None

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim * 2)
        ) if exists(time_emb_dim) else None
        self.class_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(class_emb_dim, dim * 2)
        ) if exists(class_emb_dim) else None
        self.cond_conv = SkipConv1D(in_channels=dim, out_channels=dim * 2)
        # build blocks
        self.blocks = nn.ModuleList([
            CondBlock(dim=dim, input_resolution=input_resolution,
                        num_heads=num_heads, window_size=window_size,
                        shift_size=0 if (
                            i % 2 == 0) else window_size // 2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(
                            drop_path, list) else drop_path,
                        norm_layer=norm_layer,
                        pretrained_window_size=pretrained_window_size)
            for i in range(depth)])
        
    def forward(self, x, time_emb=None, cond_emb=None, class_emb=None):
        scale_shift_list = []
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b 1 c')
            scale_shift = time_emb.chunk(2, dim = 2)
            scale_shift_list.append(scale_shift)
        if exists(cond_emb):
            cond_emb = self.cond_conv(cond_emb)
            cond_scale_shift = cond_emb.chunk(2, dim = 2)
            scale_shift_list.append(cond_scale_shift)
        if exists(self.class_mlp) and exists(class_emb):
            class_emb = self.class_mlp(class_emb)
            class_emb = rearrange(class_emb, 'b c -> b 1 c')
            class_scale_shift = class_emb.chunk(2, dim = 2)
            scale_shift_list.append(class_scale_shift)

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
        for idx, blk in enumerate(self.blocks):
            if idx == 0:
                blk_scale_shift = scale_shift_list
            else:
                blk_scale_shift = None
            if self.use_checkpoint:
                x = checkpoint(blk, x, blk_scale_shift,
                               use_reentrant=False)
            else:
                x = blk(x, blk_scale_shift)
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
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)

class AttnLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, upsample=None,
                 use_checkpoint=False, pretrained_window_size=0, time_emb_dim=None, full_attn=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
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
            self.downsample = None
        if upsample is not None:
            self.upsample = upsample(input_resolution,
                                     dim=dim, norm_layer=norm_layer)
            dim //= 2
            num_heads = max(num_heads // 2, 1)
            window_size *= 2
            input_resolution = [input_resolution[0] * 2,
                                input_resolution[1] * 2]
        else:
            self.upsample = None

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim * 2)
        ) if exists(time_emb_dim) else None

        # build blocks
        self.blocks = nn.ModuleList([
            AttnBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (
                                     i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(
                                     drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size,
                                 full_attn=full_attn)
            for i in range(depth)])
        
    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b 1 c')
            scale_shift = time_emb.chunk(2, dim = 2)

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
        for idx, blk in enumerate(self.blocks):
            if idx == 0:
                blk_scale_shift = scale_shift
            else:
                blk_scale_shift = None
            if self.use_checkpoint:
                x = checkpoint(blk, x, blk_scale_shift,
                               use_reentrant=False)
            else:
                x = blk(x, blk_scale_shift)
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
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)

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



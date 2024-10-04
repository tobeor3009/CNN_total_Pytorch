
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from timm.models.layers import DropPath, to_3tuple
import numpy as np
from ..layers import get_act, get_norm
from ..layers import PixelShuffle3D
from .swin_layers import DEFAULT_ACT, DROPOUT_INPLACE
from ..model_3d.swin_layers import window_partition, window_reverse
from ..model_3d.swin_layers import Mlp, WindowAttention
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

class PixelShuffleLinear(nn.Module):
    def __init__(self, upscale_factor):
        '''
        :param scale: upsample scale
        '''
        super().__init__()

        if isinstance(upscale_factor, int):
            upscale_factor = (upscale_factor, upscale_factor, upscale_factor)
        self.scale_num = np.prod(upscale_factor)

    def forward(self, input):
        batch_size, component, channels = input.size()
        nOut = channels // self.scale_num

        out_component = component * self.scale_num

        input_view = input.view(batch_size, component, self.scale_num, nOut)

        output = input_view.permute(0, 2, 1, 3)
        output = output.contiguous()

        return output.view(batch_size, out_component, nOut)

class SkipConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, norm, kernel_size=1):
        super().__init__()
        self.skip_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, 
                                   kernel_size=kernel_size, stride=1, padding="same")
        self.norm = norm(out_channels)
    def forward(self, *args):
        # expected shape: [B, N, C]
        x = torch.cat(args, dim=2)
        x = self.skip_conv(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        return x
    
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

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_3tuple(self.window_size),
                                    num_heads=num_heads, qkv_bias=qkv_bias,
                                    qkv_drop=qkv_drop, attn_drop=attn_drop, proj_drop=drop,
                                    pretrained_window_size=to_3tuple(pretrained_window_size))

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            Z, H, W = self.input_resolution
            img_mask = torch.zeros((1, Z, H, W, 1))  # 1 H W 1
            z_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for z in z_slices:
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, z, h, w, :] = cnt
                        cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1,
                                             self.window_size ** 3)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, scale_shift_list=None):
        Z, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == Z * H * W, f"input feature has wrong size, {L} != {Z}, {H}, {W}"

        shortcut = x
        x = x.view(B, Z, H, W, C)
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x,
                                   shifts=(-self.shift_size,
                                           -self.shift_size,
                                           -self.shift_size),
                                   dims=(1, 2, 3))
        else:
            shifted_x = x

        # partition windows
        # nW*B, window_size, window_size, C
        x_windows = window_partition(shifted_x, self.window_size)
        # nW*B, window_size**3, C
        x_windows = x_windows.view(-1, self.window_size ** 3, C)

        # W-MSA/SW-MSA
        # nW*B, window_size**3, C
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1,
                                         self.window_size,
                                         self.window_size,
                                         self.window_size, C)
        shifted_x = window_reverse(attn_windows,
                                   self.window_size, Z, H, W)  # B Z' H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x,
                           shifts=(self.shift_size,
                                   self.shift_size,
                                   self.shift_size), dims=(1, 2, 3))
        else:
            x = shifted_x
        x = x.view(B, Z * H * W, C)
        norm = self.norm1(x)
        if exists(scale_shift_list):
            for scale_shift in scale_shift_list:
                scale, shift = scale_shift
                norm = norm * (scale + 1) + shift
        x = shortcut + self.drop_path(norm)
        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

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
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        patches_resolution = np.array([img_size[0] // patch_size[0],
                                       img_size[1] // patch_size[1],
                                       img_size[2] // patch_size[2]])
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = np.prod(patches_resolution)

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, Z, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert Z == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2], \
            f"Input simage size ({Z}*{H}*{W}) doesn't match model ({np.prod(self.img_size)})."        
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Pz*Ph*Pw C
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
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: B, Z*H*W, C
        """
        Z, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == Z * H * W, f"input feature has wrong size {L} != {H}, {W}"
        assert Z % 2 == 0 and H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, Z, H, W, C)

        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, 0::2, :]
        x4 = x[:, 0::2, 0::2, 1::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3,
                       x4, x5, x6, x7], dim=-1)  # B Z/2 H/2 W/2 8*C
        x = x.view(B, -1, 8 * C)  # B Z/2*H/2*W/2 8*C

        x = self.reduction(x)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        Z, H, W = self.input_resolution
        flops = (Z // 2) * (H // 2) * (W // 2) * 8 * self.dim * 2 * self.dim
        flops += Z * H * W * self.dim // 2
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

        pixel_conv = nn.Conv3d(dim, dim * (dim_scale ** 3) // 2,
                               kernel_size=1, padding=0, bias=False)
        if dim_scale == 1:
            self.pixel_shuffle = pixel_conv
        else:
            self.pixel_shuffle = nn.Sequential(
                            pixel_conv,
                            PixelShuffle3D(dim_scale)
            )
        self.norm_layer = norm_layer(dim // 2)

    def forward(self, x):
        Z, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == Z * H * \
            W, f"input feature has wrong size {L} != {Z}, {H}, {W}"
        assert Z % 2 == 0 and H % 2 == 0 and W % 2 == 0, f"x size ({Z}*{H}*{W}) are not even."
        x = x.permute(0, 2, 1).view(B, C, Z, H, W)
        x = self.pixel_shuffle(x)
        x = x.permute(0, 2, 3, 4, 1).view(B, -1, self.dim // 2)
        x = self.norm_layer(x)
        if not self.return_vector:
            x = x.permute(0, 2, 1).view(B, self.dim // 2,
                                        Z * self.dim_scale,
                                        H * self.dim_scale,
                                        W * self.dim_scale)
        return x

class PatchExpandingLinear(nn.Module):
    def __init__(self, input_resolution, dim,
                 return_vector=True, dim_scale=2,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.return_vector = return_vector
        self.dim_scale = dim_scale
        
        assert dim_scale in [1, 2], "not supported dim_scale"
        self.mlp_layer = nn.Linear(dim, dim * (dim_scale ** 3) // 2, bias=False)
        self.norm_layer = norm_layer(dim // 2)

    def forward(self, x):
        Z, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == Z * H * \
            W, f"input feature has wrong size {L} != {Z}, {H}, {W}"
        assert Z % 2 == 0 and H % 2 == 0 and W % 2 == 0, f"x size ({Z}*{H}*{W}) are not even."

        x = self.mlp_layer(x)
        if self.dim_scale == 2:
            half_dim = C // 2
            x_expand = torch.empty(B, 2 * Z, 2 * H, 2 * W, half_dim, device=x.device, dtype=x.dtype)
            x = x.reshape(B, Z, H, W, 4 * C)

            x_expand[:, 0::2, 0::2, 0::2, :] = x[:, :, :, :, :half_dim]  # B H/2 W/2 C
            x_expand[:, 1::2, 0::2, 0::2, :] = x[:, :, :, :, half_dim:half_dim * 2]  # B H/2 W/2 C
            x_expand[:, 0::2, 1::2, 0::2, :] = x[:, :, :, :, half_dim * 2:half_dim * 3]  # B H/2 W/2 C
            x_expand[:, 1::2, 1::2, 0::2, :] = x[:, :, :, :, half_dim * 3:half_dim * 4]  # B H/2 W/2 C
            x_expand[:, 0::2, 0::2, 1::2, :] = x[:, :, :, :, half_dim * 4:half_dim * 5]  # B H/2 W/2 C
            x_expand[:, 1::2, 0::2, 1::2, :] = x[:, :, :, :, half_dim * 5:half_dim * 6]  # B H/2 W/2 C
            x_expand[:, 0::2, 1::2, 1::2, :] = x[:, :, :, :, half_dim * 6:half_dim * 7]  # B H/2 W/2 C
            x_expand[:, 1::2, 1::2, 1::2, :] = x[:, :, :, :, half_dim * 7:half_dim * 8]  # B H/2 W/2 C
            x = x_expand.view(B, 8 * Z * H * W, half_dim) # B 4*C H/2 W/2
        x = self.norm_layer(x)
        if not self.return_vector:
            x = x.permute(0, 2, 1).view(B, self.dim // 2,
                                        Z * self.dim_scale,
                                        H * self.dim_scale,
                                        W * self.dim_scale)
        return x
    
class PatchExpandingMulti(nn.Module):
    def __init__(self, input_resolution, dim,
                 return_vector=True, dim_scale=2,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.return_vector = return_vector
        self.dim_scale = dim_scale
        pixel_shuffle_conv_1 = nn.Conv3d(dim, dim * (dim_scale ** 3) // 2,
                                         kernel_size=1, padding=0, bias=False)
        pixel_shuffle = PixelShuffle3D(dim_scale)
        pixel_shuffle_conv_2 = nn.Conv3d(dim // 2, dim // 2,
                                         kernel_size=1, padding=0, bias=False)
        self.pixel_shuffle_layer = nn.Sequential(
            pixel_shuffle_conv_1,
            pixel_shuffle,
            pixel_shuffle_conv_2
        )
        upsample_layer = nn.Upsample(scale_factor=dim_scale, mode='trilinear')
        conv_after_upsample = nn.Conv3d(dim, dim // 2, kernel_size=1)
        self.upsample_layer = nn.Sequential(
            upsample_layer,
            conv_after_upsample
        )
        self.concat_conv = nn.Conv3d(dim, dim // 2, kernel_size=3, padding=1, stride=1)
        
        self.norm_layer = norm_layer(dim // 2)

    def forward(self, x):
        Z, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == Z * H * W, f"input feature has wrong size {L} != {Z}, {H}, {W}"
        assert Z % 2 == 0 and H % 2 == 0 and W % 2 == 0, f"x size ({Z}*{H}*{W}) are not even."
        x = x.permute(0, 2, 1).view(B, C, Z, H, W)
        pixel_shuffle = self.pixel_shuffle_layer(x)
        upsample = self.upsample_layer(x)
        x = torch.cat([pixel_shuffle, upsample], dim=1)
        x = self.concat_conv(x)
        
        x = x.permute(0, 2, 3, 1).view(B, -1, self.dim // 2)
        x = self.norm_layer(x)
        if not self.return_vector:
            x = x.permute(0, 2, 1).view(B, self.dim // 2,
                                        Z * self.dim_scale,
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
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, upsample=None,
                 use_checkpoint=False, pretrained_window_size=0,
                 emb_dim_list=[], use_residual=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.depth = depth
        self.use_checkpoint = use_checkpoint

        emb_block_list = []
        for emb_dim in emb_dim_list:
            emb_block = nn.Sequential(
                                        nn.SiLU(),
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
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size,
                                 use_residual=use_residual)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution,
                                         dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
        if upsample is not None:
            self.upsample = upsample(input_resolution,
                                     dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None
        
    def forward(self, x, *args):
        scale_shift_list = []
        for emb_block, emb in zip(self.emb_block_list, args):
            emb = emb_block(emb)
            if emb.ndim == 2:
                emb = emb.unsqueeze(1)
            scale_shift = emb.chunk(2, dim=2)
            scale_shift_list.append(scale_shift)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint(blk, x, scale_shift_list,
                               use_reentrant=False)
            else:
                x = blk(x, scale_shift_list)
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
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.weight, 0)
            if hasattr(blk.norm1, "bias"):
                nn.init.constant_(blk.norm1.bias, 0)
            if hasattr(blk.norm2, "bias"):
                nn.init.constant_(blk.norm2.bias, 0)


class BasicLayerV2(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., qkv_drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, upsample=None,
                 use_checkpoint=False, pretrained_window_size=0,
                 emb_dim_list=[], use_residual=False):

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
                                input_resolution[1] // 2,
                                input_resolution[2] // 2]
        else:
            self.downsample = None
        if upsample is not None:
            self.upsample = upsample(input_resolution,
                                     dim=dim, norm_layer=norm_layer)
            dim //= 2
            num_heads = max(num_heads // 2, 1)
            window_size *= 2
            input_resolution = [input_resolution[0] * 2,
                                input_resolution[1] * 2,
                                input_resolution[2] * 2]
        else:
            self.upsample = None

        emb_block_list = []
        for emb_dim in emb_dim_list:
            emb_block = nn.Sequential(
                                        nn.SiLU(),
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
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size,
                                 use_residual=use_residual)
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
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.weight, 0)
            if hasattr(blk.norm1, "bias"):
                nn.init.constant_(blk.norm1.bias, 0)
            if hasattr(blk.norm2, "bias"):
                nn.init.constant_(blk.norm2.bias, 0)

class BaseBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding='same',
                 norm=nn.GroupNorm, groups=1, act=DEFAULT_ACT, bias=False):
        super().__init__()

        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
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

class ConvBlock3D(nn.Module):
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
            elif emb_type == "3d":
                emb_block = BaseBlock3D(emb_dim, out_channels * 2, kernel_size,
                                        1, padding, norm, groups, act, bias)
            else:
                raise Exception("emb_type must be seq or 3d")
            emb_block_list.append(emb_block)
        self.emb_block_list = nn.ModuleList(emb_block_list)

        self.block_1 = BaseBlock3D(in_channels, out_channels, kernel_size,
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
            emb = rearrange(emb, 'b c -> b c 1 1 1')
            scale_shift = emb.chunk(2, dim=1)
            scale_shift_list.append(scale_shift)
        x = self.block_1(x, scale_shift_list)
        return x
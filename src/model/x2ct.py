import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange

from timm.models.layers import DropPath, to_3tuple, trunc_normal_

from .swin_transformer_unet_skip_expand_decoder_sys import PatchEmbed, BasicLayer, PatchMerging, Mlp


def window_partition_3d(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, Z, H, W, C = x.shape
    x = x.view(B,
               Z // window_size, window_size,
               H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous(
    ).view(-1, window_size, window_size, window_size, C)
    return windows


def window_reverse_3d(windows, window_size, Z, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (Z * H * W / (window_size ** 3)))
    x = windows.view(B,
                     Z // window_size, H // window_size, W // window_size,
                     window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(Z, B, H, W, -1)
    return x


class PatchExpand3D(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 4 * dim,
                                bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        Z, H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W * Z, f"input feature has wrong size {L} != {H}, {W}"

        x = x.view(B, Z, H, W, C)
        x = rearrange(x, 'b z h w (p1 p2 p3 c)-> b (z p1) (h p2) (w p3) c',
                      p1=2, p2=2, c=C // 8)
        x = x.view(B, -1, C // 8)
        x = self.norm(x)

        return x


class WindowAttention3D(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_z = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(
            [coords_z, coords_h, coords_w]))  # 3, Wz, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wz*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]  # 3, Wz*Wh*Ww, Wz*Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wz*Wh*Ww, Wz*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= ((2 * self.window_size[1] - 1) *
                                     (2 * self.window_size[2] - 1))
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)

        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        window_total_size = np.prod(self.window_size)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            window_total_size, window_total_size, -1)  # Wz*Wh*Ww,Wz*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wz*Wh*Ww, Wz*Wh*Ww
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
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

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


class SwinTransformerBlock3D(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
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

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=to_3tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

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

            # nW, window_size, window_size, window_size, 1
            mask_windows = window_partition_3d(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1,
                                             self.window_size ** 3)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        Z, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == Z * H * W, f"input feature has wrong size {L} != {H}, {W}"

        shortcut = x
        x = self.norm1(x)
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
        x_windows = window_partition_3d(shifted_x, self.window_size)
        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size ** 3, C)

        # W-MSA/SW-MSA
        # nW*B, window_size*window_size, C
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1,
                                         self.window_size, self.window_size, self.window_size, C)
        shifted_x = window_reverse_3d(
            attn_windows, self.window_size, Z, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x,
                           shifts=(self.shift_size,
                                   self.shift_size,
                                   self.shift_size),
                           dims=(1, 2, 3))
        else:
            x = shifted_x
        x = x.view(B, Z * H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        Z, H, W = self.input_resolution
        # norm1
        flops += self.dim * Z * H * W
        # W-MSA/SW-MSA
        nW = Z * H * W / (self.window_size ** 3)
        flops += nW * self.attn.flops(self.window_size ** 3)
        # mlp
        flops += 2 * Z * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * Z * H * W
        return flops


class BasicLayer_up3D(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(dim=dim, input_resolution=input_resolution,
                                   num_heads=num_heads, window_size=window_size,
                                   shift_size=0 if (
                                       i % 2 == 0) else window_size // 2,
                                   mlp_ratio=mlp_ratio,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   drop=drop, attn_drop=attn_drop,
                                   drop_path=drop_path[i] if isinstance(
                                       drop_path, list) else drop_path,
                                   norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand3D(input_resolution,
                                          dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class FinalPatchExpand_X43D(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, (dim_scale ** 3) * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        Z, H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == Z * H * W, f"input feature has wrong size {L} != {H}, {W}"

        x = x.view(B, Z, H, W, C)
        x = rearrange(x, 'b z h w (p1 p2 p3 c)-> b (z p1) (h p2) (w p3) c',
                      p1=self.dim_scale,
                      p2=self.dim_scale,
                      p3=self.dim_scale,
                      c=C // (self.dim_scale**3))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class PatchExpanding_2D_3D(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2,
                 norm_layer=nn.LayerNorm, preserve_dim=False, return_vector=False):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.return_vector = return_vector
        total_downsample = int(math.log(input_resolution[0], 2))
        expand_num = total_downsample // 2
        is_odd = total_downsample % 2
        expand_num += is_odd
        self.downsample_scale_list = []
        self.expand_list = nn.ModuleList()
        self.norm_list = nn.ModuleList()

        for idx in range(expand_num):
            if idx < expand_num - 1:
                down_scale = 4
                expand_dim = 4 * dim
                new_dim = dim
            else:
                if is_odd == 1:
                    down_scale = 2
                    expand_dim = dim
                else:
                    down_scale = 4
                    expand_dim = 2 * dim
                new_dim = dim // dim_scale
                if preserve_dim:
                    expand_dim *= 2
                    new_dim *= 2
            expand = nn.Linear(dim, expand_dim,
                               bias=False) if dim_scale == 2 else nn.Identity()
            norm = norm_layer(new_dim)
            self.downsample_scale_list.append(down_scale)
            self.expand_list.append(expand)
            self.norm_list.append(norm)

    def forward(self, x):

        H, W = self.input_resolution
        Z = H

        for down_scale, expand, norm in zip(self.downsample_scale_list, self.expand_list, self.norm_list):
            x = expand(x)
            B, L, C = x.shape
            x = torch.reshape(x, (B, -1, H, W, C))
            x = rearrange(x, 'b z h w (p1 c)-> b (z p1) h w c',
                          p1=down_scale, c=C // down_scale)
            x = torch.reshape(x, (B, -1, C // down_scale))
            x = norm(x)
        if self.return_vector:
            pass
        else:
            x = torch.reshape(x, (B, Z, H, W, C // down_scale))

        return x


class PatchExpand3D(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 4 * dim,
                                bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        Z, H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W * Z, f"input feature has wrong size {L} != {H}, {W}"

        x = x.view(B, Z, H, W, C)
        x = rearrange(x, 'b z h w (p1 p2 p3 c)-> b (z p1) (h p2) (w p3) c',
                      p1=2, p2=2, p3=2, c=C // 8)
        x = x.view(B, -1, C // 8)
        x = self.norm(x)

        return x


class X2CTSwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], depths_decoder=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        print("SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(depths,
                                                                                                                         depths_decoder, drop_path_rate, num_classes))
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim,
                                      norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(
                                   depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (
                                   i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.x2ct_layer = PatchExpanding_2D_3D(input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                                 patches_resolution[1] // (2 ** i_layer)),
                                               dim=int(
                                                   embed_dim * 2 ** i_layer),
                                               preserve_dim=True,
                                               return_vector=True)
        self.skip_x2ct_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            skip_x2ct = PatchExpanding_2D_3D(input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                               patches_resolution[1] // (2 ** i_layer)),
                                             dim=int(embed_dim * 2 ** i_layer),
                                             preserve_dim=True,
                                             return_vector=True)
            self.skip_x2ct_layers.append(skip_x2ct)
        # build decoder layers
        self.ct_layers = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dim_param = 2 ** (self.num_layers - 1 - i_layer)
            concat_linear = nn.Linear(2 * int(embed_dim * dim_param),
                                      int(embed_dim * dim_param)) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand3D(input_resolution=(patches_resolution[0] // dim_param,
                                                           patches_resolution[0] // dim_param,
                                                           patches_resolution[0] // dim_param),
                                         dim=int(embed_dim * dim_param),
                                         dim_scale=2,
                                         norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up3D(dim=int(embed_dim * dim_param),
                                           input_resolution=(patches_resolution[0] // dim_param,
                                                             patches_resolution[0] // dim_param,
                                                             patches_resolution[0] // dim_param),
                                           depth=depths[(
                                               self.num_layers - 1 - i_layer)],
                                           num_heads=num_heads[(
                                               self.num_layers - 1 - i_layer)],
                                           window_size=window_size,
                                           mlp_ratio=self.mlp_ratio,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop=drop_rate, attn_drop=attn_drop_rate,
                                           drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                               depths[:(self.num_layers - 1 - i_layer) + 1])],
                                           norm_layer=norm_layer,
                                           upsample=PatchExpand3D if (
                    i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint)
            self.ct_layers.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X43D(input_resolution=(img_size // patch_size,
                                                              img_size // patch_size,
                                                              img_size // patch_size),
                                            dim_scale=patch_size, dim=embed_dim)
            self.output = nn.Conv3d(in_channels=embed_dim, out_channels=self.num_classes,
                                    kernel_size=1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)  # B L C

        return x, x_downsample

    # Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.ct_layers):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B L C

        return x

    def up_x4(self, x):
        Z, H, W = self.patches_resolution[0], self.patches_resolution[0], self.patches_resolution[0]
        B, L, C = x.shape
        assert L == Z * H * W, "input size unmatched"
        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B,
                       self.patch_size * Z,
                       self.patch_size * H,
                       self.patch_size * W, -1)
            x = x.permute(0, 4, 1, 2, 3)  # B, C, Z, H, W
            x = self.output(x)

        return x

    def forward(self, x):
        x, x_downsample_list = self.forward_features(x)
        x = self.x2ct_layer(x)
        skip_x_downsample_list = []
        for skip_x2ct, x_downsample in zip(self.skip_x2ct_layers, x_downsample_list):
            skip_x_downsample = skip_x2ct(x_downsample)
            skip_x_downsample_list.append(skip_x_downsample)
        x = self.forward_up_features(x, skip_x_downsample_list)
        x = self.up_x4(x)

        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * \
            self.patches_resolution[0] * \
            self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

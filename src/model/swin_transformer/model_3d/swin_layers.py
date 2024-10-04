import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from timm.models.layers import DropPath, to_3tuple, trunc_normal_
import numpy as np
from einops import rearrange
from ..model_2d.swin_layers import ChannelDropout, Mlp
from ..layers import get_act, PixelShuffle3D
DEFAULT_ACT = get_act("leakyrelu")
DROPOUT_INPLACE = False

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, window_size, C)
    """
    B, Z, H, W, C = x.shape
    x = x.view(B, Z // window_size, window_size,
               H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous(
    ).view(-1, window_size, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, Z, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        Z (int): Dim of image
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (Z * H * W / (window_size ** 3)))
    x = windows.view(B, Z // window_size, H // window_size, W // window_size,
                     window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Z, H, W, -1)
    return x


def get_len(x):
    if hasattr(x, "shape"):
        x_len = x.shape[0]
    else:
        x_len = len(x)
    return x_len


def meshgrid_3d(*arrs):
    arrs = tuple(reversed(arrs))  # edit
    lens = list(map(get_len, arrs))
    dim = len(arrs)

    sz = 1
    for s in lens:
        sz *= s

    ans = []
    for i, arr in enumerate(arrs):
        slc = [1] * dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j != i:
                arr2 = arr2.repeat(sz, axis=j)
        ans.append(arr2)

    return tuple(ans)


class WindowAttention(nn.Module):
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

        logit_scale_param = torch.log(10 * torch.ones((num_heads, 1, 1)))
        self.logit_scale = nn.Parameter(logit_scale_param, requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(3, cbp_dim, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(cbp_dim, num_heads, bias=False))
        # get relative_coords_table
        relative_coords_z = np.arange(-(self.window_size[0] - 1),
                                      self.window_size[0], dtype=np.float32)
        relative_coords_h = np.arange(-(self.window_size[1] - 1),
                                      self.window_size[1], dtype=np.float32)
        relative_coords_w = np.arange(-(self.window_size[2] - 1),
                                      self.window_size[2], dtype=np.float32)
        relative_coords_table = meshgrid_3d(relative_coords_z,
                                            relative_coords_h,
                                            relative_coords_w)
        relative_coords_table = np.stack(relative_coords_table, axis=0)
        relative_coords_table = torch.tensor(relative_coords_table,
                                             dtype=torch.float32)
        relative_coords_table = relative_coords_table.permute(
            1, 2, 3, 0).contiguous().unsqueeze(0)  # 1, 2*Wz-1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[..., 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[..., 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[..., 0] /= (self.window_size[0] - 1)
            relative_coords_table[..., 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_z = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = meshgrid_3d(coords_z,
                             coords_h,
                             coords_w)
        coords = np.stack(coords, axis=0)  # 3, Wz, Wh, Ww
        coords = torch.tensor(coords, dtype=torch.float32)
        coords_flatten = torch.flatten(coords, 1)  # 3, Wz*Wh*Ww
        relative_coords = (coords_flatten[:, :, None] -
                           coords_flatten[:, None, :])  # 3, Wz*Wh*Ww, Wz*Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wz*Wh*Ww, Wz*Wh*Ww, 3
        # Adjusting the relative positions
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= ((2 * self.window_size[1] - 1) *
                                     (2 * self.window_size[2] - 1))
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        # Wz*Wh*Ww, Wz*Wh*Ww
        relative_position_index = relative_coords.sum(-1).long()

        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.qkv_drop = ChannelDropout(qkv_drop, inplace=DROPOUT_INPLACE)
        self.attn_drop = nn.Dropout(attn_drop, inplace=DROPOUT_INPLACE)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = ChannelDropout(proj_drop, inplace=DROPOUT_INPLACE)
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
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(
            -1)]
        relative_position_bias = relative_position_bias.view(np.prod(self.window_size),
                                                             np.prod(self.window_size), -1)  # Wz*Wh*Ww,Wz*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wz*Wh*Ww, Wz*Wh*Ww
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

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_3tuple(self.window_size),
                                    num_heads=num_heads, qkv_bias=qkv_bias,
                                    qkv_drop=qkv_drop, attn_drop=attn_drop, proj_drop=drop,
                                    pretrained_window_size=to_3tuple(pretrained_window_size),
                                    cbp_dim=np.clip(np.prod(input_resolution) // 16, 256, 1024))

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

    def forward(self, x):
        Z, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == Z * H * W, f"input feature has wrong size {L} != {Z}, {H}, {W}"

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
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

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
        nW = Z * H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * Z * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * Z * H * W
        return flops


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
        self.pixel_shuffle_conv_1 = nn.Conv3d(dim, dim * (dim_scale ** 3) // 2,
                                              kernel_size=1, padding=0, bias=False)
        self.pixel_shuffle = PixelShuffle3D(dim_scale)
        self.norm_layer = norm_layer(dim // 2)

    def forward(self, x):
        Z, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == Z * H * \
            W, f"input feature has wrong size {L} != {Z}, {H}, {W}"
        assert Z % 2 == 0 and H % 2 == 0 and W % 2 == 0, f"x size ({Z}*{H}*{W}) are not even."
        x = x.permute(0, 2, 1).view(B, C, Z, H, W)
        x = self.pixel_shuffle_conv_1(x)
        x = self.pixel_shuffle(x)
        x = x.permute(0, 2, 3, 4, 1).view(B, -1, self.dim // 2)
        x = self.norm_layer(x)
        if not self.return_vector:
            x = x.permute(0, 2, 1).view(B, self.dim // 2,
                                        Z * self.dim_scale,
                                        H * self.dim_scale,
                                        W * self.dim_scale)
        return x


class PatchExpandingConcat(nn.Module):
    def __init__(self, input_resolution, dim,
                 return_vector=True, dim_scale=2,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.return_vector = return_vector
        self.dim_scale = dim_scale
        self.pixel_shuffle_conv = nn.Conv3d(dim, dim * (dim_scale ** 3) // 2,
                                            kernel_size=1, padding=0, bias=False)
        self.pixel_shuffle = PixelShuffle3D(dim_scale)
        self.upsample = nn.Upsample(scale_factor=(dim_scale, dim_scale, dim_scale),
                                    mode='trilinear')
        self.upsample_conv = nn.Conv3d(dim, dim // 2,
                                       kernel_size=3, padding=1, bias=False)
        self.cat_conv = nn.Conv3d(dim, dim // 2,
                                  kernel_size=1, padding=0, bias=False)
        self.norm_layer = norm_layer(dim // 2)

    def forward(self, x):
        Z, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == Z * H * \
            W, f"input feature has wrong size {L} != {Z}, {H}, {W}"
        assert Z % 2 == 0 and H % 2 == 0 and W % 2 == 0, f"x size ({Z}*{H}*{W}) are not even."
        x = x.permute(0, 2, 1).view(B, C, Z, H, W)
        pixel_shuffle = self.pixel_shuffle_conv(x)
        pixel_shuffle = self.pixel_shuffle(pixel_shuffle)
        upsample = self.upsample(x)
        upsample = self.upsample_conv(upsample)
        x = torch.cat([pixel_shuffle, upsample], dim=1)
        x = self.cat_conv(x)
        x = x.permute(0, 2, 3, 4, 1).view(B, -1, self.dim // 2)
        x = self.norm_layer(x)
        if not self.return_vector:
            x = x.permute(0, 2, 1).view(B, self.dim // 2,
                                        Z * self.dim_scale,
                                        H * self.dim_scale,
                                        W * self.dim_scale)
        return x


class BasicLayerV1(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., qkv_drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, upsample=None,
                 use_checkpoint=False, pretrained_window_size=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (
                                     i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, qkv_drop=qkv_drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(
                                     drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
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

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint(blk, x,
                               use_reentrant=False)

            else:
                x = blk(x)
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
                 mlp_ratio=4., qkv_bias=True, drop=0., qkv_drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, upsample=None,
                 use_checkpoint=False, pretrained_window_size=0):

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
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (
                                     i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, qkv_drop=qkv_drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(
                                     drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

    def forward(self, x):
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
                x = checkpoint(blk, x,
                               use_reentrant=False)
            else:
                x = blk(x)
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
        Zo, Ho, Wo = self.patches_resolution
        flops = Zo * Ho * Wo * self.embed_dim * \
            self.in_chans * np.prod(self.patch_size)
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

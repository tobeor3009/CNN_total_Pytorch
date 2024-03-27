import torch
from torch import nn, einsum
from torch.nn.utils import spectral_norm
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from ..common_module.cbam import CBAM
from ..common_module.layers import get_act, get_norm, DEFAULT_ACT
from ..common_module.layers import ConcatBlock
import math
import numpy as np
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

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

class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dim_head=None):
        super().__init__()
        self.num_heads = num_heads
        if dim_head is None:
            dim_head = dim
        self.dim_head = dim_head
        self.scale = (dim_head / num_heads) ** -0.5
        self.prenorm = LayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim_head * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(dim_head, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.prenorm(x)
        qkv = self.to_qkv(x)
        qkv = qkv.chunk(3, dim = 1)          
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.num_heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.num_heads, x = h, y = w)
        return self.to_out(out) + x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, dim_head=None):
        super().__init__()
        self.num_heads = num_heads
        if dim_head is None:
            dim_head = dim

        self.dim_head = dim_head
        self.scale = (dim_head / num_heads) ** -0.5

        self.prenorm = LayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim_head * 3, 1, bias = False)
        self.to_out = nn.Conv2d(dim_head, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.prenorm(x)

        qkv = self.to_qkv(x)
        qkv = qkv.chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.num_heads), qkv)
        q = q * self.scale
        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out) + x
    
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
    
class BaseBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding='same',
                 norm="batch", groups=1, act=DEFAULT_ACT, bias=False):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups, bias=bias)
        if norm == "spectral":
            self.conv = spectral_norm(self.conv)
        if not bias or norm != "spectral":
            self.norm_layer = get_norm(norm, out_channels, mode="2d")
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

# attn_info.keys = ["emb_type_list", "num_heads", "full_attn"]
class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding='same',
                 norm="batch", groups=1, act=DEFAULT_ACT, bias=False,
                 emb_dim_list=None, attn_info=None, use_checkpoint=False):
        super().__init__()

        assert emb_dim_list is not None, f"You need to set emb_dim_list. current emb_dim_list: {emb_dim_list}"
        self.attn_info = attn_info
        self.use_checkpoint = use_checkpoint
        # you always have time embedding
        emb_type_list = ["seq"]
        if exists(attn_info):
            emb_type_list += attn_info["emb_type_list"]
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
        self.block_2 = BaseBlock2D(out_channels, out_channels, kernel_size,
                                    1, padding, norm, groups, act, bias)

        if in_channels != out_channels or stride != 1:
            self.residiual_conv = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        else:
            self.residiual_conv = nn.Identity()

        if attn_info is None:
            self.attn = nn.Identity()
        elif attn_info["full_attn"]:
            self.attn = Attention(out_channels, attn_info["num_heads"])
        else:
            self.attn = LinearAttention(out_channels, attn_info["num_heads"])

    def forward(self, x, *args):
        if self.use_checkpoint:
            return checkpoint(self._forward_impl, x, *args,
                              use_reentrant=False)
        else:
            return self._forward_impl(x, *args)

    def _forward_impl(self, x, *args):
        skip_x = x
        scale_shift_list = []
        for emb_block, emb in zip(self.emb_block_list, args):
            emb = emb_block(emb)
            if emb.ndim == 2:
                emb = rearrange(emb, 'b c -> b c 1 1')
            scale_shift = emb.chunk(2, dim=1)
            scale_shift_list.append(scale_shift)
        x = self.block_1(x, scale_shift_list)
        x = self.block_2(x)
        x = x + self.residiual_conv(skip_x)
        x = self.attn(x)
        return x
    
class Inception_Resnet_Block2D(nn.Module):
    def __init__(self, in_channels, scale, block_type, block_size=16,
                 include_cbam=True, norm="batch", act=DEFAULT_ACT, 
                 emb_dim_list=None, attn_info=None, use_checkpoint=False):
        super().__init__()
        assert emb_dim_list is not None, f"You need to set emb_dim_list. current emb_dim_list: {emb_dim_list}"
        self.include_cbam = include_cbam
        self.scale = scale
        if block_type == 'block35':
            branch_0 = ConvBlock2D(in_channels, block_size * 2, 1,
                                   norm=norm, act=act,
                                   emb_dim_list=emb_dim_list, attn_info=attn_info,
                                   use_checkpoint=use_checkpoint)
            branch_1 = MultiInputSequential(
                ConvBlock2D(in_channels, block_size * 2, 1,
                            norm=norm, act=act,
                            emb_dim_list=emb_dim_list, attn_info=attn_info, 
                            use_checkpoint=use_checkpoint),
                ConvBlock2D(block_size * 2, block_size * 2, 3,
                            norm=norm, act=act,
                            emb_dim_list=emb_dim_list, attn_info=attn_info,
                            use_checkpoint=use_checkpoint)
            )
            branch_2 = MultiInputSequential(
                ConvBlock2D(in_channels, block_size * 2, 1,
                            norm=norm, act=act, 
                            emb_dim_list=emb_dim_list, attn_info=attn_info,
                            use_checkpoint=use_checkpoint),
                ConvBlock2D(block_size * 2, block_size * 3, 3,
                            norm=norm, act=act,
                            emb_dim_list=emb_dim_list, attn_info=attn_info,
                            use_checkpoint=use_checkpoint),
                ConvBlock2D(block_size * 3, block_size * 4, 3,
                            norm=norm, act=act,
                            emb_dim_list=emb_dim_list, attn_info=attn_info,
                            use_checkpoint=use_checkpoint)
            )
            mixed_channel = block_size * 8
            # block_size * (2 + 2 + 4) => block_size * 10
            branches = [branch_0, branch_1, branch_2]
        elif block_type == 'block17':
            branch_0 = ConvBlock2D(in_channels, block_size * 12, 1,
                                   norm=norm, act=act,
                                   emb_dim_list=emb_dim_list, attn_info=attn_info,
                                   use_checkpoint=use_checkpoint)
            branch_1 = MultiInputSequential(
                ConvBlock2D(in_channels, block_size * 8, 1,
                            norm=norm, act=act,
                            emb_dim_list=emb_dim_list, attn_info=attn_info,
                            use_checkpoint=use_checkpoint),
                ConvBlock2D(block_size * 8, block_size * 10, [1, 7],
                            norm=norm, act=act,
                            emb_dim_list=emb_dim_list, attn_info=attn_info,
                            use_checkpoint=use_checkpoint),
                ConvBlock2D(block_size * 10, block_size * 12, [7, 1],
                            norm=norm, act=act,
                            emb_dim_list=emb_dim_list, attn_info=attn_info,
                            use_checkpoint=use_checkpoint)
            )
            mixed_channel = block_size * 24
            branches = [branch_0, branch_1]
        elif block_type == 'block8':
            branch_0 = ConvBlock2D(in_channels, block_size * 12, 1,
                                   norm=norm, act=act,
                                   emb_dim_list=emb_dim_list, attn_info=attn_info,
                                   use_checkpoint=use_checkpoint)
            branch_1 = MultiInputSequential(
                ConvBlock2D(in_channels, block_size * 12, 1,
                            norm=norm, act=act,
                            emb_dim_list=emb_dim_list, attn_info=attn_info,
                            use_checkpoint=use_checkpoint),
                ConvBlock2D(block_size * 12, block_size * 14, [1, 3],
                            norm=norm, act=act,
                            emb_dim_list=emb_dim_list, attn_info=attn_info,
                            use_checkpoint=use_checkpoint),
                ConvBlock2D(block_size * 14, block_size * 16, [3, 1],
                            norm=norm, act=act,
                            emb_dim_list=emb_dim_list, attn_info=attn_info,
                            use_checkpoint=use_checkpoint)
            )
            mixed_channel = block_size * 28
            branches = [branch_0, branch_1]
        else:
            raise ValueError('Unknown Inception-ResNet block type. '
                             'Expects "block35", "block17" or "block8", '
                             'but got: ' + str(block_type))
        self.mixed = ConcatBlock(branches)
        # TBD: Name?
        self.up = ConvBlock2D(mixed_channel, in_channels, 1,
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

class MultiDecoder2D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 norm="layer", act=DEFAULT_ACT, kernel_size=2,
                 emb_dim_list=None, attn_info=None, use_checkpoint=False):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        conv_before_pixel_shuffle = nn.Conv2d(in_channels=in_channels,
                                              out_channels=in_channels *
                                              np.prod(kernel_size),
                                              kernel_size=1)
        pixel_shuffle_layer = nn.PixelShuffle(upscale_factor=(kernel_size
                                                              if isinstance(kernel_size, int)
                                                              else kernel_size[0]))
        conv_after_pixel_shuffle = nn.Conv2d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=1)
        self.pixel_shuffle = nn.Sequential(
            conv_before_pixel_shuffle,
            pixel_shuffle_layer,
            conv_after_pixel_shuffle
        )
        upsample_layer = nn.Upsample(scale_factor=kernel_size,
                                        mode='bilinear')
        conv_after_upsample = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=1)
        self.upsample = nn.Sequential(
            upsample_layer,
            conv_after_upsample
        )
        self.concat_conv = ConvBlock2D(in_channels=out_channels * 2,
                                       out_channels=out_channels, kernel_size=3,
                                       stride=1, padding=1, norm=norm, act=act,
                                       emb_dim_list=emb_dim_list, attn_info=attn_info,
                                       use_checkpoint=use_checkpoint)

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
    
class Output2D(nn.Module):
    def __init__(self, in_channels, out_channels, act=None):
        super().__init__()
        conv_out_channels = in_channels // 2
        self.conv_5x5 = nn.Conv2d(in_channels=in_channels,
                                  out_channels=conv_out_channels,
                                  kernel_size=5, padding=2)
        self.conv_3x3 = nn.Conv2d(in_channels=in_channels,
                                  out_channels=conv_out_channels,
                                  kernel_size=3, padding=1)
        self.concat_conv = nn.Conv2d(in_channels=conv_out_channels * 2,
                                        out_channels=out_channels,
                                        kernel_size=3, padding=1)
        self.act = get_act(act)

    def forward(self, x):
        conv_5x5 = self.conv_5x5(x)
        conv_3x3 = self.conv_3x3(x)
        output = torch.cat([conv_5x5, conv_3x3], dim=1)
        output = self.concat_conv(output)
        output = self.act(output)
        return output
    
class MaxPool2d(nn.MaxPool2d):
    def forward(self, x, *args):
        return super().forward(x)
class AvgPool2d(nn.AvgPool2d):
    def forward(self, x, *args):
        return super().forward(x)
class MultiInputSequential(nn.Sequential):
    def forward(self, x, *args):
        for module in self:
            x = module(x, *args)
        return x
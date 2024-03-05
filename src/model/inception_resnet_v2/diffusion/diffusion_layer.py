import torch
from torch import nn
from torch.nn.utils import spectral_norm
from einops import rearrange, einsum
from ..common_module.cbam import CBAM
from ..common_module.layers import get_act, get_norm, DEFAULT_ACT
from ..common_module.layers import LambdaLayer, ConcatBlock
import math

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
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.prenorm = LayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.prenorm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.prenorm = LayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.prenorm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)
    
class SinusoidalPosEmb(Module):
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
    
class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding='same',
                 norm="batch", groups=1, act=DEFAULT_ACT, bias=False, time_emb_channels=None):
        super().__init__()

        if exists(time_emb_channels):
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_channels, out_channels * 2)
            )
            if in_channels != out_channels or stride == 2:
                self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            else: 
                self.res_conv = nn.Identity()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups,  bias=bias)
        if norm == "spectral":
            self.conv = spectral_norm(self.conv)
        if not bias or norm != "spectral":
            self.norm_layer = get_norm(norm, out_channels, mode="2d")
        else:
            self.norm_layer = nn.Identity()
        self.act_layer = get_act(act)

    def forward(self, x, time_emb=None):
        conv = self.conv(x)
        norm = self.norm_layer(conv)

        if exists(self.mlp):
            assert time_emb is not None, "time_emb is None with expected use"
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)
            scale, shift = scale_shift
            norm = norm * (scale + 1) + shift
            act = self.act_layer(norm)
            return act + self.res_conv(x)
        else:
            act = self.act_layer(norm)
            return act


class Inception_Resnet_Block2D(nn.Module):
    def __init__(self, in_channels, scale, block_type, block_size=16,
                 include_cbam=True, norm="batch", act=DEFAULT_ACT):
        super().__init__()
        self.include_cbam = include_cbam
        if block_type == 'block35':
            branch_0 = ConvBlock2D(in_channels, block_size * 2, 1,
                                   norm=norm, act=act)
            branch_1 = nn.Sequential(
                ConvBlock2D(in_channels, block_size * 2, 1,
                            norm=norm, act=act),
                ConvBlock2D(block_size * 2, block_size * 2, 3,
                            norm=norm, act=act)
            )
            branch_2 = nn.Sequential(
                ConvBlock2D(in_channels, block_size * 2, 1,
                            norm=norm, act=act),
                ConvBlock2D(block_size * 2, block_size * 3, 3,
                            norm=norm, act=act),
                ConvBlock2D(block_size * 3, block_size * 4, 3,
                            norm=norm, act=act)
            )
            mixed_channel = block_size * 8
            branches = [branch_0, branch_1, branch_2]
        elif block_type == 'block17':
            branch_0 = ConvBlock2D(in_channels, block_size * 12, 1,
                                   norm=norm, act=act)
            branch_1 = nn.Sequential(
                ConvBlock2D(in_channels, block_size * 8, 1,
                            norm=norm, act=act),
                ConvBlock2D(block_size * 8, block_size * 10, [1, 7],
                            norm=norm, act=act),
                ConvBlock2D(block_size * 10, block_size * 12, [7, 1],
                            norm=norm, act=act)
            )
            mixed_channel = block_size * 24
            branches = [branch_0, branch_1]
        elif block_type == 'block8':
            branch_0 = ConvBlock2D(in_channels, block_size * 12, 1,
                                   norm=norm, act=act)
            branch_1 = nn.Sequential(
                ConvBlock2D(in_channels, block_size * 12, 1,
                            norm=norm, act=act),
                ConvBlock2D(block_size * 12, block_size * 14, [1, 3],
                            norm=norm, act=act),
                ConvBlock2D(block_size * 14, block_size * 16, [3, 1],
                            norm=norm, act=act)
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
                              bias=True, norm=norm, act=None)
        if self.include_cbam:
            self.cbam = CBAM(gate_channels=in_channels,
                             reduction_ratio=16)
        # TBD: implement of include_context
        self.residual_add = LambdaLayer(
            lambda inputs: inputs[0] + inputs[1] * scale)
        self.act = get_act(act)

    def forward(self, x):
        mixed = self.mixed(x)
        up = self.up(mixed)
        if self.include_cbam:
            up = self.cbam(up)
        residual_add = self.residual_add([x, up])
        act = self.act(residual_add)
        return act



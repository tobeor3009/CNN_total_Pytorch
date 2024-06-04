import math
from functools import partial, wraps

import torch
from torch import sqrt
from torch import nn, einsum
import torch.nn.functional as F
from torch.special import expm1
from torch.cuda.amp import autocast

import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange
from torch.utils.checkpoint import checkpoint

def process_layer(use_checkpoint, layer, x, *emb_list):
    if use_checkpoint:
        x = checkpoint(layer, x, *emb_list,
                        use_reentrant=False)
    else:
        x = layer(x, *emb_list)
    return x

# helpers
def exists(val):
    return val is not None

def identity(t):
    return t

def is_lambda(f):
    return callable(f) and f.__name__ == "<lambda>"

def default(val, d):
    if exists(val):
        return val
    return d() if is_lambda(d) else d

def cast_tuple(t, l = 1):
    return ((t,) * l) if not isinstance(t, tuple) else t

def append_dims(t, dims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * dims))

def l2norm(t):
    return F.normalize(t, dim = -1)

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

        input_view = input.view(batch_size, nOut, self.scale[0], self.scale[1],
                                self.scale[2], in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4)
        output = output.contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)
    
# u-vit related functions and modules
class Upsample(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        factor = 2,
        mode="2d"
    ):
        super().__init__()
        assert mode in ["2d", "3d"], 'mode not in ["2d", 3d"]'
        self.factor = factor

        dim_out = default(dim_out, dim)
        if mode == "2d":
            self.factor_squared = factor ** 2
            conv = nn.Conv2d(dim, dim_out * self.factor_squared, 1)
            pixel_shuffle = nn.PixelShuffle(factor)
        elif mode == "3d":
            self.factor_squared = factor ** 3
            conv = nn.Conv3d(dim, dim_out * self.factor_squared, 1)
            pixel_shuffle = PixelShuffle3D(factor)
        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            pixel_shuffle
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, *kernel_shape = conv.weight.shape
        conv_weight = torch.empty(o // self.factor_squared, i, *kernel_shape)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r = self.factor_squared)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)

def Downsample(
    dim,
    dim_out = None,
    factor = 2,
    mode = "2d"
):
    assert mode in ["2d", "3d"], 'mode not in ["2d", 3d"]'
    
    if mode =="2d":
        return nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=factor, p2=factor),
            nn.Conv2d(dim * (factor ** 2), default(dim_out, dim), 1)
        )
    elif mode == "3d":
        return nn.Sequential(
            Rearrange('b c (z p1) (h p2) (w p3) -> b (c p1 p2 p3) z h w', p1=factor, p2=factor, p3=factor),
            nn.Conv3d(dim * (factor ** 3), default(dim_out, dim), 1)
        )

class RMSNorm(nn.Module):
    def __init__(self, dim, scale = True, normalize_dim = 2):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim)) if scale else 1

        self.scale = scale
        self.normalize_dim = normalize_dim

    def forward(self, x):
        normalize_dim = self.normalize_dim
        scale = append_dims(self.g, x.ndim - self.normalize_dim - 1) if self.scale else 1
        return F.normalize(x, dim = normalize_dim) * scale * (x.shape[normalize_dim] ** 0.5)

# sinusoidal positional embeds

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

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, mode="2d", use_checkpoint=False):
        super().__init__()
        if mode == "2d":
            self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        elif mode == "3d":
            self.proj = nn.Conv3d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out, normalize_dim = 1)
        self.act = nn.SiLU()
        
        self.use_checkpoint = use_checkpoint
    
    def forward(self, x, scale_shift_list=[]):
        x = process_layer(self.use_checkpoint, self._forward_impl, x, scale_shift_list)
        return x
    
    def _forward_impl(self, x, scale_shift_list=[]):
        x = self.proj(x)
        x = self.norm(x)

        for scale_shift in scale_shift_list:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, emb_dim_list=[], mode="2d", use_checkpoint=False):
        super().__init__()
        assert mode in ["2d", "3d"], 'mode not in ["2d", 3d"]'
        self.use_checkpoint = use_checkpoint
        self.emb_mlp_list = nn.ModuleList()
        self.mode = mode
        for emb_dim in emb_dim_list:
            emb_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_dim, dim_out * 2)
            )
            self.emb_mlp_list.append(emb_mlp)

        self.block1 = Block(dim, dim_out, mode=mode, use_checkpoint=use_checkpoint)
        self.block2 = Block(dim_out, dim_out, mode=mode, use_checkpoint=use_checkpoint)
        if mode == "2d":
            self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        elif mode == "3d":
            self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        
    def forward(self, x, *emb_list):

        scale_shift_list = []
        for emb_mlp, emb in zip(self.emb_mlp_list, emb_list):
            emb = emb_mlp(emb)
            if self.mode == "2d":
                emb = rearrange(emb, 'b c -> b c 1 1')
            elif self.mode == "3d":
                emb = rearrange(emb, 'b c -> b c 1 1 1')
            scale_shift = emb.chunk(2, dim = 1)
            scale_shift_list.append(scale_shift)
        
        h = self.block1(x, scale_shift_list=scale_shift_list)
        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, mode="2d", use_checkpoint=False):
        super().__init__()
        assert mode in ["2d", "3d"], 'mode not in ["2d", 3d"]'
        self.use_checkpoint = use_checkpoint
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        if mode == "2d":
            conv_layer = nn.Conv2d
            self.to_seq = lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads)
            self.to_img = lambda out, h, w: rearrange(out, 'b h c (x y) -> b (h c) x y',
                                                      h=self.heads, x=h, y=w)
        elif mode == "3d":
            conv_layer = nn.Conv3d
            self.to_seq = lambda t: rearrange(t, 'b (h c) z x y -> b h c (z x y)', h=self.heads)
            self.to_img = lambda out, z, h, w: rearrange(out, 'b h c (z x y) -> b (h c) z x y',
                                                         h=self.heads, z=z, x=h, y=w)
        self.norm = RMSNorm(dim, normalize_dim=1)
        self.to_qkv = conv_layer(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            conv_layer(hidden_dim, dim, 1),
            RMSNorm(dim, normalize_dim = 1)
        )


    def forward(self, x):
        x = process_layer(self.use_checkpoint, self._forward_impl, x)
        return x
    
    def _forward_impl(self, x):
        residual = x

        b, c, *size = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(self.to_seq, qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = self.to_img(out, *size)

        return self.to_out(out) + residual

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, scale = 8, dropout = 0., use_checkpoint = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

    def forward(self, x):
        x = process_layer(self.use_checkpoint, self._forward_impl, x)
        return x
        
    def _forward_impl(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        q, k = map(l2norm, (q, k))

        q = q * self.q_scale
        k = k * self.k_scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        emb_dim_list=[],
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.norm = RMSNorm(dim, scale = False)
        dim_hidden = dim * mult

        self.emb_mlp_list = nn.ModuleList()
        for emb_dim in emb_dim_list:
            emb_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_dim, dim_hidden * 2),
                Rearrange('b d -> b 1 d')
            )
            to_scale_shift_linear = emb_mlp[-2]
            nn.init.zeros_(to_scale_shift_linear.weight)
            nn.init.zeros_(to_scale_shift_linear.bias)

            self.emb_mlp_list.append(emb_mlp)

        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim_hidden, bias = False),
            nn.SiLU()
        )

        self.proj_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim, bias = False)
        )

    def forward(self, x, *emb_list):
        x = self.norm(x)
        x = self.proj_in(x)

        for emb_mlp, emb in zip(self.emb_mlp_list, emb_list):
            scale, shift = emb_mlp(emb).chunk(2, dim = -1)
            x = x * (scale + 1) + shift

        return self.proj_out(x)

# vit

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        emb_dim_list,
        depth,
        dim_head = 32,
        heads = 4,
        ff_mult = 4,
        dropout = 0.,
        use_checkpoint=False
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = dropout),
                FeedForward(dim = dim, mult = ff_mult, emb_dim_list = emb_dim_list, dropout = dropout)
            ]))

    def forward(self, x, *emb_list):
        for attn, ff in self.layers:
            if self.use_checkpoint:
                x = checkpoint(self.process_block, attn, ff, x, *emb_list,
                              use_reentrant=False)
            else:
                x = self.process_block(attn, ff, x, *emb_list)
        return x
    
    def process_block(self, attn, ff, x, *emb_list):
        x = attn(x) + x
        x = ff(x, *emb_list) + x
        return x
    
class AttentionPool(nn.Module):
    def __init__(self, feature_num: tuple, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(feature_num + 1,
                                                             embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(
            2, 0, 1)  # BC(HW) -> NBC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (N+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (N+1)NC
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
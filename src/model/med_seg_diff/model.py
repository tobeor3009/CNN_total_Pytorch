import math
import copy
from random import random
from functools import partial
from collections import namedtuple

from beartype import beartype

import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.fft import fft2, ifft2

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm

# constants

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class LayerNorm(Module):
    def __init__(self, dim, bias = False):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1)) if bias else None

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g + default(self.b, 0)

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

# building block modules

class Block(Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv2d(dim, inner_dim, 1),
        nn.GELU(),
        nn.Conv2d(inner_dim, dim, 1),
    )

class LinearAttention(Module):
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

class Attention(Module):
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

class Transformer(Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        heads = 4,
        depth = 1
    ):
        super().__init__()
        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(ModuleList([
                Residual(Attention(dim, dim_head = dim_head, heads = heads)),
                Residual(FeedForward(dim))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x

# vision transformer for dynamic ff-parser

class ViT(Module):
    def __init__(
        self,
        dim,
        *,
        image_size,
        patch_size,
        channels = 3,
        channels_out = None,
        dim_head = 32,
        heads = 4,
        depth = 4,
    ):
        super().__init__()
        assert exists(image_size)
        assert (image_size % patch_size) == 0

        num_patches_height_width = image_size // patch_size

        self.pos_emb = nn.Parameter(torch.zeros(dim, num_patches_height_width, num_patches_height_width))

        channels_out = default(channels_out, channels)

        patch_dim = channels * (patch_size ** 2)
        output_patch_dim = channels_out * (patch_size ** 2)

        self.to_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = patch_size, p2 = patch_size),
            nn.Conv2d(patch_dim, dim, 1),
            LayerNorm(dim)
        )

        self.transformer = Transformer(
            dim = dim,
            dim_head = dim_head,
            depth = depth
        )

        self.to_patches = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, output_patch_dim, 1),
            Rearrange('b (c p1 p2) h w -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size),
        )

        nn.init.zeros_(self.to_patches[-2].weight)
        nn.init.zeros_(self.to_patches[-2].bias)

    def forward(self, x):
        x = self.to_tokens(x)
        x = x + self.pos_emb

        x = self.transformer(x)
        return self.to_patches(x)

# conditioning class

class Conditioning(Module):
    def __init__(
        self,
        fmap_size,
        dim,
        dynamic = True,
        image_size = None,
        dim_head = 32,
        heads = 4,
        depth = 4,
        patch_size = 16
    ):
        super().__init__()
        self.ff_parser_attn_map = nn.Parameter(torch.ones(dim, fmap_size, fmap_size))

        self.dynamic = dynamic

        if dynamic:
            self.to_dynamic_ff_parser_attn_map = ViT(
                dim = dim,
                channels = dim * 2 * 2,  # both input and condition, and account for complex (real and imag components)
                channels_out = dim,
                image_size = image_size,
                patch_size = patch_size,
                heads = heads,
                dim_head = dim_head
            )

        self.norm_input = LayerNorm(dim, bias = True)
        self.norm_condition = LayerNorm(dim, bias = True)

        self.block = ResnetBlock(dim, dim)

    def forward(self, x, c):
        ff_parser_attn_map = self.ff_parser_attn_map

        # ff-parser in the paper, for modulating out the high frequencies

        dtype = x.dtype
        x = fft2(x)

        if self.dynamic:
            c_complex = fft2(c)
            x_as_real, c_as_real = map(torch.view_as_real, (x, c_complex))
            x_as_real, c_as_real = map(lambda t: rearrange(t, 'b d h w ri -> b (d ri) h w'), (x_as_real, c_as_real))

            to_dynamic_input = torch.cat((x_as_real, c_as_real), dim = 1)

            dynamic_ff_parser_attn_map = self.to_dynamic_ff_parser_attn_map(to_dynamic_input)

            ff_parser_attn_map = ff_parser_attn_map + dynamic_ff_parser_attn_map

        x = x * ff_parser_attn_map

        x = ifft2(x).real
        x = x.type(dtype)

        # eq 3 in paper

        normed_x = self.norm_input(x)
        normed_c = self.norm_condition(c)
        c = (normed_x * normed_c) * c

        # add an extra block to allow for more integration of information
        # there is a downsample right after the Condition block (but maybe theres a better place to condition than right before the downsample)

        return self.block(c)

# model

@beartype
class Unet(Module):
    def __init__(
        self,
        dim,
        image_size,
        mask_channels = 1,
        input_img_channels = 3,
        init_dim = None,
        out_dim = None,
        dim_mults: tuple = (1, 2, 4, 8),
        full_self_attn: tuple = (False, False, False, True),
        attn_dim_head = 32,
        attn_heads = 4,
        mid_transformer_depth = 1,
        self_condition = False,
        resnet_block_groups = 8,
        conditioning_klass = Conditioning,
        skip_connect_condition_fmaps = False,    # whether to concatenate the conditioning fmaps in the latter decoder upsampling portion of unet
        dynamic_ff_parser_attn_map = False,      # allow for ff-parser to be dynamic based on the input. will exclude condition for now
        conditioning_kwargs: dict = dict(
            dim_head = 32,
            heads = 4,
            depth = 4,
            patch_size = 16
        )
    ):
        super().__init__()

        self.image_size = image_size

        # determine dimensions

        self.input_img_channels = input_img_channels
        self.mask_channels = mask_channels
        self.self_condition = self_condition

        output_channels = mask_channels
        mask_channels = mask_channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(mask_channels, init_dim, 7, padding = 3)
        self.cond_init_conv = nn.Conv2d(input_img_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention related params

        attn_kwargs = dict(
            dim_head = attn_dim_head,
            heads = attn_heads
        )

        # conditioner settings

        if conditioning_klass == Conditioning:
            conditioning_klass = partial(
                Conditioning,
                dynamic = dynamic_ff_parser_attn_map,
                **conditioning_kwargs
            )

        # layers

        num_resolutions = len(in_out)
        assert len(full_self_attn) == num_resolutions

        self.conditioners = ModuleList([])

        self.skip_connect_condition_fmaps = skip_connect_condition_fmaps

        # downsampling encoding blocks

        self.downs = ModuleList([])

        curr_fmap_size = image_size

        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(in_out, full_self_attn)):
            is_last = ind >= (num_resolutions - 1)
            attn_klass = Attention if full_attn else LinearAttention

            self.conditioners.append(conditioning_klass(curr_fmap_size, dim_in, image_size = curr_fmap_size))

            self.downs.append(ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(attn_klass(dim_in, **attn_kwargs)),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

            if not is_last:
                curr_fmap_size //= 2

        # middle blocks

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_transformer = Transformer(mid_dim, depth = mid_transformer_depth, **attn_kwargs)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        # condition encoding path will be the same as the main encoding path

        self.cond_downs = copy.deepcopy(self.downs)
        self.cond_mid_block1 = copy.deepcopy(self.mid_block1)

        # upsampling decoding blocks

        self.ups = ModuleList([])

        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(reversed(in_out), reversed(full_self_attn))):
            is_last = ind == (len(in_out) - 1)
            attn_klass = Attention if full_attn else LinearAttention

            skip_connect_dim = dim_in * (2 if self.skip_connect_condition_fmaps else 1)

            self.ups.append(ModuleList([
                block_klass(dim_out + skip_connect_dim, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + skip_connect_dim, dim_out, time_emb_dim = time_dim),
                Residual(attn_klass(dim_out, **attn_kwargs)),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        # projection out to predictions

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, output_channels, 1)

    def forward(
        self,
        x,
        time,
        cond,
        x_self_cond = None
    ):
        dtype, skip_connect_c = x.dtype, self.skip_connect_condition_fmaps

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        c = self.cond_init_conv(cond)

        t = self.time_mlp(time)

        h = []

        for (block1, block2, attn, downsample), (cond_block1, cond_block2, cond_attn, cond_downsample), conditioner in zip(self.downs, self.cond_downs, self.conditioners):
            x = block1(x, t)
            c = cond_block1(c, t)

            h.append([x, c] if skip_connect_c else [x])

            x = block2(x, t)
            c = cond_block2(c, t)

            x = attn(x)
            c = cond_attn(c)

            # condition using modulation of fourier frequencies with attentive map
            # you can test your own conditioners by passing in a different conditioner_klass , if you believe you can best the paper

            c = conditioner(x, c)

            h.append([x, c] if skip_connect_c else [x])

            x = downsample(x)
            c = cond_downsample(c)

        x = self.mid_block1(x, t)
        c = self.cond_mid_block1(c, t)

        x = x + c  # seems like they summed the encoded condition to the encoded input representation

        x = self.mid_transformer(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, *h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, *h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

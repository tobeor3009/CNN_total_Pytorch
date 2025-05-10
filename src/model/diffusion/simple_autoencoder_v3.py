import math
from functools import partial, wraps

import torch
from torch import sqrt
from torch import nn, einsum
import torch.nn.functional as F
from torch.special import expm1
from torch.cuda.amp import autocast

from tqdm import tqdm
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange
from torch.utils.checkpoint import checkpoint


from .utils import exists, identity, is_lambda, default, cast_tuple, append_dims, l2norm
from .utils import Upsample, Downsample, LearnedSinusoidalPosEmb, Block, ResnetBlock
from .utils import LinearAttention, Attention, FeedForward, Transformer, AttentionPool

    
# model
class UViT(nn.Module):
    def __init__(
        self,
        dim,
        img_size,
        init_dim = None,
        cond_dim = None,
        out_dim = None,
        latent_dim = None,
        dim_mults = (1, 2, 4, 8),
        downsample_factor = 2,
        channels = 3,
        vit_depth = 6,
        vit_dropout = 0.2,
        attn_dim_head = 32,
        attn_heads = 4,
        ff_mult = 4,
        learned_sinusoidal_dim = 16,
        init_img_transform: callable = None,
        final_img_itransform: callable = None,
        patch_size = 1,
        dual_patchnorm = False,
        self_condition = False,
        use_checkpoint = False
    ):
        super().__init__()
        if isinstance(use_checkpoint, bool):
            use_checkpoint = [use_checkpoint for _ in dim_mults]

        layer_depth = len(dim_mults)
        # for compability with Medsegdiff
        self.image_size = img_size
        self.input_img_channels = cond_dim
        self.mask_channels = channels
        self.self_condition = self_condition
        # for initial dwt transform (or whatever transform researcher wants to try here)

        if exists(init_img_transform) and exists(final_img_itransform):
            init_shape = torch.Size(1, 1, 32, 32)
            mock_tensor = torch.randn(init_shape)
            assert final_img_itransform(init_img_transform(mock_tensor)).shape == init_shape

        self.init_img_transform = default(init_img_transform, identity)
        self.final_img_itransform = default(final_img_itransform, identity)

        input_channels = channels

        init_dim = default(init_dim, dim)

        self.latent_encoder = LatentEncoder(dim=dim,
                                            img_size=img_size,
                                            init_dim=init_dim,
                                            latent_dim=latent_dim,
                                            dim_mults=dim_mults,
                                            downsample_factor=downsample_factor,
                                            channels=cond_dim,
                                            vit_depth=vit_depth,
                                            vit_dropout=vit_dropout,
                                            attn_dim_head=attn_dim_head,
                                            attn_heads=attn_heads,
                                            ff_mult=ff_mult,
                                            init_img_transform=init_img_transform,
                                            final_img_itransform=final_img_itransform,
                                            patch_size=patch_size,
                                            dual_patchnorm=dual_patchnorm,
                                            use_checkpoint=use_checkpoint)

        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        # whether to do initial patching, as alternative to dwt

        self.unpatchify = identity

        input_channels = channels * (patch_size ** 2)
        needs_patch = patch_size > 1

        if needs_patch:
            if not dual_patchnorm:
                self.init_conv = nn.Conv2d(channels, init_dim, patch_size, stride = patch_size)
            else:
                self.init_conv = nn.Sequential(
                    Rearrange('b c (h p1) (w p2) -> b h w (c p1 p2)', p1 = patch_size, p2 = patch_size),
                    nn.LayerNorm(input_channels),
                    nn.Linear(input_channels, init_dim),
                    nn.LayerNorm(init_dim),
                    Rearrange('b h w c -> b c h w')
                )

            self.unpatchify = nn.ConvTranspose2d(input_channels, channels, patch_size, stride = patch_size)

        # determine dimensions

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # downsample factors

        downsample_factor = cast_tuple(downsample_factor, len(dim_mults))
        assert len(downsample_factor) == len(dim_mults)

        emb_dim_list = [time_dim, latent_dim]
        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), factor) in enumerate(zip(in_out, downsample_factor)):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, emb_dim_list=emb_dim_list, use_checkpoint=use_checkpoint[ind]),
                ResnetBlock(dim_in, dim_in, emb_dim_list=emb_dim_list, use_checkpoint=use_checkpoint[ind]),
                LinearAttention(dim_in, use_checkpoint=use_checkpoint[ind]),
                Downsample(dim_in, dim_out, factor = factor)
            ]))

        mid_dim = dims[-1]

        self.vit = Transformer(
            dim = mid_dim,
            emb_dim_list = emb_dim_list,
            depth = vit_depth,
            dim_head = attn_dim_head,
            heads = attn_heads,
            ff_mult = ff_mult,
            dropout = vit_dropout,
            use_checkpoint=use_checkpoint[-1]
        )

        for ind, ((dim_in, dim_out), factor) in enumerate(zip(reversed(in_out), reversed(downsample_factor))):
            is_last = ind == (len(in_out) - 1)
            up_ind = layer_depth - ind - 1
            self.ups.append(nn.ModuleList([
                Upsample(dim_out, dim_in, factor = factor),
                ResnetBlock(dim_in * 2, dim_in, emb_dim_list=emb_dim_list, use_checkpoint=use_checkpoint[up_ind]),
                ResnetBlock(dim_in * 2, dim_in, emb_dim_list=emb_dim_list, use_checkpoint=use_checkpoint[up_ind]),
                LinearAttention(dim_in, use_checkpoint=use_checkpoint[up_ind]),
            ]))

        default_out_dim = input_channels
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = ResnetBlock(dim * 2, dim, emb_dim_list=emb_dim_list)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, cond=None, x_self_cond=None, class_labels=None,
                cond_drop_prob=None):
        
        x = self.init_img_transform(x)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)
        latent = self.latent_encoder(cond)
        emb_list = [t, latent]
        
        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, *emb_list)
            h.append(x)

            x = block2(x, *emb_list)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = rearrange(x, 'b c h w -> b h w c')
        x, ps = pack([x], 'b * c')

        x = self.vit(x, *emb_list)

        x, = unpack(x, ps, 'b * c')
        x = rearrange(x, 'b h w c -> b c h w')

        for upsample, block1, block2, attn in self.ups:
            x = upsample(x)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, *emb_list)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, *emb_list)
            x = attn(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, *emb_list)
        x = self.final_conv(x)

        x = self.unpatchify(x)
        return self.final_img_itransform(x)

class LatentEncoder(nn.Module):
    def __init__(self,
                dim,
                img_size,
                init_dim = None,
                latent_dim = None,
                dim_mults = (1, 2, 4, 8),
                downsample_factor = 2,
                channels = 3,
                vit_depth = 6,
                vit_dropout = 0.2,
                attn_dim_head = 32,
                attn_heads = 4,
                ff_mult = 4,
                init_img_transform: callable = None,
                final_img_itransform: callable = None,
                patch_size = 1,
                dual_patchnorm = False,
                use_checkpoint = False):
        super().__init__()
        if isinstance(use_checkpoint, bool):
            use_checkpoint = [use_checkpoint for _ in dim_mults]

        if exists(init_img_transform) and exists(final_img_itransform):
            init_shape = torch.Size(1, 1, 32, 32)
            mock_tensor = torch.randn(init_shape)
            assert final_img_itransform(init_img_transform(mock_tensor)).shape == init_shape

        self.init_img_transform = default(init_img_transform, identity)
        self.final_img_itransform = default(final_img_itransform, identity)

        self.init_img_transform = default(init_img_transform, identity)
        self.final_img_itransform = default(final_img_itransform, identity)

        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        input_channels = channels * (patch_size ** 2)
        needs_patch = patch_size > 1

        if needs_patch:
            if not dual_patchnorm:
                self.init_conv = nn.Conv2d(channels, init_dim, patch_size, stride = patch_size)
            else:
                self.init_conv = nn.Sequential(
                    Rearrange('b c (h p1) (w p2) -> b h w (c p1 p2)', p1 = patch_size, p2 = patch_size),
                    nn.LayerNorm(input_channels),
                    nn.Linear(input_channels, init_dim),
                    nn.LayerNorm(init_dim),
                    Rearrange('b h w c -> b c h w')
                )

            self.unpatchify = nn.ConvTranspose2d(input_channels, channels, patch_size, stride = patch_size)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        downsample_factor = cast_tuple(downsample_factor, len(dim_mults))
        assert len(downsample_factor) == len(dim_mults)

        emb_dim_list = []
        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), factor) in enumerate(zip(in_out, downsample_factor)):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, emb_dim_list=emb_dim_list, use_checkpoint=use_checkpoint[ind]),
                ResnetBlock(dim_in, dim_in, emb_dim_list=emb_dim_list, use_checkpoint=use_checkpoint[ind]),
                LinearAttention(dim_in, use_checkpoint=use_checkpoint[ind]),
                Downsample(dim_in, dim_out, factor = factor)
            ]))

        mid_dim = dims[-1]

        self.vit = Transformer(
            dim = mid_dim,
            emb_dim_list = emb_dim_list,
            depth = vit_depth,
            dim_head = attn_dim_head,
            heads = attn_heads,
            ff_mult = ff_mult,
            dropout = vit_dropout,
            use_checkpoint=use_checkpoint[-1]
        )
        feature_num = (img_size // (2 ** 4)) ** 2
        self.pool_layer = AttentionPool(feature_num=feature_num,
                                        embed_dim=mid_dim,
                                        num_heads=attn_heads, output_dim=latent_dim)
        self.out = nn.Sequential(nn.SiLU(), nn.Dropout(0.05), nn.Linear(latent_dim, latent_dim),
                                 nn.SiLU(), nn.Dropout(0.05), nn.Linear(latent_dim, latent_dim))
        
    def forward(self, x):
        x = self.init_img_transform(x)

        x = self.init_conv(x)

        for block1, block2, attn, downsample in self.downs:
            x = block1(x)
            x = block2(x)
            x = attn(x)

            x = downsample(x)

        x = rearrange(x, 'b c h w -> b h w c')
        x, ps = pack([x], 'b * c')

        x = self.vit(x)

        x, = unpack(x, ps, 'b * c')
        x = rearrange(x, 'b h w c -> b c h w')

        x = self.pool_layer(x)
        x = self.out(x)
        return x
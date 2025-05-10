import torch
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat
from functools import partial
from torch.nn import functional as F

import numpy as np
import math

DEFAULT_ACT = nn.SiLU()
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

class WrapGroupNorm(nn.GroupNorm):
    
    def __init__(self, num_channels, *args, **kwargs):
        super().__init__(num_channels=num_channels,*args, **kwargs)

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
    
class LinearAttention2D(nn.Module):
    def __init__(self, dim, num_heads=4, dim_head=None):
        super().__init__()
        self.num_heads = num_heads
        if dim_head is None:
            dim_head = dim
        self.dim_head = dim_head
        self.scale = (dim_head / num_heads) ** -0.5
        self.to_qkv = nn.Conv2d(dim, dim_head * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(dim_head, dim, 1),
            nn.GroupNorm(num_channels=dim, num_groups=num_heads)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        # x.shape = [B, C, X, Y]
        # q.shape, k.shape, v.shape = [B, H, C, N]
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.num_heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)
        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.num_heads, x = h, y = w)
        return self.to_out(out)

class Attention2D(nn.Module):
    def __init__(self, dim, num_heads=4, dim_head=None):
        super().__init__()
        self.num_heads = num_heads
        if dim_head is None:
            dim_head = dim

        self.dim_head = dim_head
        self.scale = (dim_head / num_heads) ** -0.5

        self.to_qkv = nn.Conv2d(dim, dim_head * 3, 1, bias = False)
        self.to_out = nn.Conv2d(dim_head, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.num_heads), qkv)
        
        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

class MultiDecoder2D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 norm="layer", act=DEFAULT_ACT, kernel_size=2):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        conv_before_pixel_shuffle = nn.Conv2d(in_channels=in_channels,
                                              out_channels=in_channels * np.prod(kernel_size),
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
        self.concat_conv = nn.Conv2d(in_channels=out_channels * 2,
                                        out_channels=out_channels,
                                        kernel_size=3, padding=1)
        self.norm = norm(out_channels)
        self.act = act

    def forward(self, x):
        pixel_shuffle = self.pixel_shuffle(x)
        upsample = self.upsample(x)
        out = torch.cat([pixel_shuffle, upsample], dim=1)
        out = self.concat_conv(out)
        out = self.norm(out)
        out = self.act(out)
        return out

class Output2D(nn.Module):
    def __init__(self, in_channels, out_channels):
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
    def forward(self, x):
        conv_5x5 = self.conv_5x5(x)
        conv_3x3 = self.conv_3x3(x)
        output = torch.cat([conv_5x5, conv_3x3], dim=1)
        output = self.concat_conv(output)
        return output

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
    
class BaseBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding='same',
                 norm=WrapGroupNorm, groups=1, act=DEFAULT_ACT, bias=False):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups, bias=bias)
        if not bias:
            self.norm_layer = norm(out_channels)
        else:
            self.norm_layer = nn.Identity()
        self.act_layer = act

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
                 norm=WrapGroupNorm, groups=1, act=DEFAULT_ACT, bias=False,
                 emb_dim_list=[], emb_type_list=[], use_checkpoint=False):
        super().__init__()

        self.use_checkpoint = use_checkpoint
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
            if emb.ndim == 2:
                emb = rearrange(emb, 'b c -> b c 1 1')
            scale_shift = emb.chunk(2, dim=1)
            scale_shift_list.append(scale_shift)
        x = self.block_1(x, scale_shift_list)
        return x
    
class DiffusionUnet(nn.Module):
    def __init__(self, img_shape, in_chans, cond_chans, out_chans, emb_chans,
                 num_class_embeds=None, cond_drop_prob=0.5, block_size=64, num_head_list=[2, 4, 8, 8],
                 self_condition=False, use_checkpoint=False
                 ):
        super().__init__()

        # for compability with Medsegdiff
        self.image_size = img_shape[0]
        self.input_img_channels = cond_chans
        self.mask_channels = in_chans
        #######################################

        time_emb_dim = block_size * 4
        layer_depth = len(num_head_list)

        if isinstance(use_checkpoint, bool):
            use_checkpoint = [use_checkpoint for _ in num_head_list]

        assert len(use_checkpoint) == len(num_head_list), "num_head_list and use_checkpoint len not matched"

        self.num_class_embeds = num_class_embeds
        if num_class_embeds is not None:
            self.class_emb_layer = nn.Embedding(num_class_embeds, time_emb_dim)
            if cond_drop_prob > 0:
                self.null_class_emb = nn.Parameter(torch.randn(time_emb_dim))
            else:
                self.null_class_emb = None
            
            self.class_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim)
            )
            emb_dim_list = [time_emb_dim, time_emb_dim, emb_chans]
            emb_type_list = ["seq", "seq", "seq"]
        else:
            self.class_emb_layer = None
            self.null_class_emb = None
            self.class_mlp = None
            emb_dim_list = [time_emb_dim, emb_chans]
            emb_type_list = ["seq", "seq"]

        ########################
        self.block_size = block_size
        self.time_emb_dim = time_emb_dim
        self.num_head_list = num_head_list
        self.out_chans = out_chans
        self.cond_drop_prob = cond_drop_prob
        self.layer_depth = layer_depth
        self.emb_dim_list = emb_dim_list
        self.emb_type_list = emb_type_list
        self.self_condition = self_condition
        self.use_checkpoint = use_checkpoint
        ########################
        if self_condition:
            in_chans = in_chans * 2

        self.latent_model = DiffusionEncoder(img_shape=img_shape, in_chans=cond_chans, emb_chans=emb_chans,
                                             block_size=block_size, num_head_list=num_head_list, use_checkpoint=use_checkpoint)

        self.init_block = ConvBlock2D(in_chans, block_size, 3, 1,
                                      norm=partial(WrapGroupNorm, num_groups=num_head_list[0]),
                                      emb_dim_list=emb_dim_list, emb_type_list=emb_type_list)

        self.time_mlp = self.get_time_mlp_seq()
        self.down_block_list_1, self.down_block_list_2, self.down_attn_list = self.get_encode_block_list()
        self.mid_block_1, self.mid_attn, self.mid_block_2 = self.get_mid_block_list()
        self.up_block_list_1, self.up_block_list_2, self.up_attn_list, self.up_list = self.get_decode_block_list()
        self.final_layer_1, self.final_layer_2 = self.get_final_conv_layers()
        
    def forward(self, x, time, cond=None, x_self_cond=None, class_labels=None,
                cond_drop_prob=None):
        
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        batch, device = x.size(0), x.device

        time_emb = self.time_mlp(time)
        latent = self.latent_model(cond)

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        time_emb = self.time_mlp(time)
        latent = self.latent_model(cond)

        if self.num_class_embeds is not None:
            class_emb = self.class_emb_layer(class_labels).to(dtype=x.dtype)
            if cond_drop_prob > 0:
                keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device=device)
                null_classes_emb = repeat(self.null_class_emb, 'd -> b d', b=batch)

                class_emb = torch.where(
                    rearrange(keep_mask, 'b -> b 1'),
                    class_emb, null_classes_emb
                )

            class_emb = self.class_mlp(class_emb)
            emb_list = [time_emb, class_emb, latent]
        else:
            emb_list = [time_emb, latent]
        
        x = self.init_block(x, *emb_list)

        x, skip_list_1, skip_list_2 = self.process_encode_block(x, *emb_list)
        x = self.process_mid_block(x, *emb_list)
        x = self.process_decode_block(skip_list_1, skip_list_2, x, *emb_list)

        x = self.final_layer_1(x, *emb_list)
        x = self.final_layer_2(x)
        return x
    
    def get_time_mlp_seq(self):
        time_emb_dim = self.time_emb_dim
        time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim // 2),
            nn.Linear(time_emb_dim // 2, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        return time_mlp
    
    def process_checkpoint_block(self, use_checkpoint, block, x, *emb_list):
        if use_checkpoint:
            return checkpoint(block, x, *emb_list,
                              use_reentrant=False)
        else:
            return block(x, *emb_list)

    def process_encode_block(self, x, *emb_list):
        skip_list_1 = []
        skip_list_2 = []
        for down_block_1, down_block_2, down_attn, use_checkpoint in zip(self.down_block_list_1, self.down_block_list_2, 
                                                                         self.down_attn_list, self.use_checkpoint):
            
            x = down_block_1(x, *emb_list)
            skip_list_1.append(x)
            
            x = down_block_2(x, *emb_list)
            skip_list_2.append(x)

            x = self.process_checkpoint_block(use_checkpoint, down_attn, x)

        return x, skip_list_1, skip_list_2
    
    def process_mid_block(self, x, *emb_list):
        x = self.mid_block_1(x, *emb_list)
        x = self.process_checkpoint_block(self.use_checkpoint[-1], self.mid_attn, x)
        x = self.mid_block_2(x, *emb_list)
        return x
    
    def process_decode_block(self, skip_list_1, skip_list_2, x, *emb_list):
        for up_block_1, up_block_2, up_attn, up, use_checkpoint in zip(self.up_block_list_1, self.up_block_list_2, 
                                                                       self.up_attn_list, self.up_list, self.use_checkpoint[::-1]):
            
            skip_x = skip_list_1.pop()
            x = torch.cat([x, skip_x], dim=1)
            x = up_block_1(x, *emb_list)

            skip_x = skip_list_2.pop()
            x = torch.cat([x, skip_x], dim=1)
            x = up_block_2(x, *emb_list)

            x = self.process_checkpoint_block(use_checkpoint, up_attn, x)
            x = self.process_checkpoint_block(use_checkpoint, up, x)
        return x

    def get_encode_block_list(self):
        down_block_list_1 = nn.ModuleList()
        down_block_list_2 = nn.ModuleList()
        down_attn_list = nn.ModuleList()

        for encode_idx in range(self.layer_depth):
            
            if encode_idx < 3:
                in_block_size = self.block_size * (2 ** encode_idx)
            else:
                in_block_size = self.block_size * (2 ** 3)

            if encode_idx < 2:
                out_block_size = in_block_size * 2
            else:
                out_block_size = self.block_size * (2 ** 3)
            
            num_heads = self.num_head_list[encode_idx]
            norm_fn = partial(WrapGroupNorm, num_groups=num_heads)

            common_kwarg_dict = {"kernel_size": 3, "padding":1, "norm": norm_fn,
                                 "emb_dim_list": self.emb_dim_list, "emb_type_list":self.emb_type_list,
                                 "use_checkpoint": self.use_checkpoint[encode_idx]}

            down_block_1 = ConvBlock2D(in_block_size, out_block_size, stride=2,
                                       **common_kwarg_dict)
            down_block_2 = ConvBlock2D(out_block_size, out_block_size, stride=1,
                                       **common_kwarg_dict)
            down_attn = LinearAttention2D(dim=out_block_size, num_heads=num_heads)

            down_block_list_1.append(down_block_1)
            down_block_list_2.append(down_block_2)
            down_attn_list.append(down_attn)
        return down_block_list_1, down_block_list_2, down_attn_list
    
    def get_mid_block_list(self):
        mid_block_size = self.block_size * (2 ** min(self.layer_depth, 3))
        num_heads = self.num_head_list[-1]
        norm_fn = partial(WrapGroupNorm, num_groups=num_heads)
        common_kwarg_dict = {"kernel_size": 3, "stride": 1, "norm": norm_fn,
                            "emb_dim_list": self.emb_dim_list, "emb_type_list":self.emb_type_list,
                            "use_checkpoint": self.use_checkpoint[-1]}
        mid_block_1 = ConvBlock2D(mid_block_size, mid_block_size, **common_kwarg_dict)
        mid_attn = Attention2D(dim=mid_block_size, num_heads=num_heads)
        mid_block_2 = ConvBlock2D(mid_block_size, mid_block_size, **common_kwarg_dict)

        return mid_block_1, mid_attn, mid_block_2
    
    def get_decode_block_list(self):
        up_block_list_1 = nn.ModuleList()
        up_block_list_2 = nn.ModuleList()
        up_attn_list = nn.ModuleList()
        up_list = nn.ModuleList()

        for decode_idx in range(self.layer_depth, 0, -1):

            if decode_idx > 3:
                in_block_size = self.block_size * (2 ** 4)
                mid_block_size = in_block_size // 2
                out_block_size = mid_block_size
            else:
                in_block_size = self.block_size * (2 ** (decode_idx + 1))
                mid_block_size = in_block_size // 2
                out_block_size = mid_block_size // 2

            num_heads = self.num_head_list[decode_idx - 1]
            norm_fn = partial(WrapGroupNorm, num_groups=num_heads)
            common_kwarg_dict = {"kernel_size": 3, "stride": 1, "norm": norm_fn,
                                 "emb_dim_list": self.emb_dim_list, "emb_type_list":self.emb_type_list,
                                 "use_checkpoint": self.use_checkpoint[decode_idx - 1]}
                                
            up_block_1 = ConvBlock2D(in_block_size, mid_block_size, **common_kwarg_dict)
            up_block_2 = ConvBlock2D(in_block_size, out_block_size, **common_kwarg_dict)
            up_attn = LinearAttention2D(dim=out_block_size, num_heads=num_heads)
            up = MultiDecoder2D(out_block_size, out_block_size, norm=norm_fn)

            up_block_list_1.append(up_block_1)
            up_block_list_2.append(up_block_2)
            up_attn_list.append(up_attn)
            up_list.append(up)

        return up_block_list_1, up_block_list_2, up_attn_list, up_list


    def get_final_conv_layers(self):
        num_heads = self.num_head_list[0]
        norm_fn = partial(WrapGroupNorm, num_groups=num_heads)
        common_kwarg_dict = {"kernel_size": 3, "stride": 1, "norm": norm_fn,
                            "emb_dim_list": self.emb_dim_list, "emb_type_list":self.emb_type_list,
                            "use_checkpoint": self.use_checkpoint[0]}
        final_layer_1 = ConvBlock2D(self.block_size, self.block_size, **common_kwarg_dict)
        final_layer_2 = Output2D(self.block_size, self.out_chans)
        return final_layer_1, final_layer_2

class DiffusionEncoder(DiffusionUnet):
    def __init__(self, img_shape, in_chans, emb_chans,
                 block_size=64, num_head_list=[2, 4, 8, 8], use_checkpoint=False):
        super(DiffusionUnet, self).__init__()

        time_emb_dim = block_size * 4
        layer_depth = len(num_head_list)
        if isinstance(use_checkpoint, bool):
            use_checkpoint = [use_checkpoint for _ in num_head_list]

        assert len(use_checkpoint) == len(num_head_list), "num_head_list and use_checkpoint len not matched"
        ########################
        self.block_size = block_size
        self.time_emb_dim = time_emb_dim
        self.num_head_list = num_head_list
        self.layer_depth = layer_depth
        self.emb_dim_list = []
        self.emb_type_list = []
        self.use_checkpoint = use_checkpoint
        ########################

        self.init_block = ConvBlock2D(in_chans, block_size, 3, 1,
                                      norm=partial(WrapGroupNorm, num_groups=num_head_list[0]),
                                      emb_dim_list=[], emb_type_list=[])

        self.down_block_list_1, self.down_block_list_2, self.down_attn_list = self.get_encode_block_list()
        self.mid_block_1, self.mid_attn, self.mid_block_2 = self.get_mid_block_list()

        feature_shape = np.array(img_shape) // (2 ** layer_depth)
        mid_block_size = block_size * (2 ** min(layer_depth, 3))
        self.pool_layer = AttentionPool(feature_num=np.prod(feature_shape),
                                        embed_dim=mid_block_size,
                                        num_heads=num_head_list[-1], output_dim=emb_chans * 2)
        self.out = nn.Sequential(nn.SiLU(), nn.Dropout(0.05), nn.Linear(emb_chans * 2, emb_chans),
                                 nn.SiLU(), nn.Dropout(0.05), nn.Linear(emb_chans, emb_chans))
        
    def forward(self, x):
        
        x = self.init_block(x)

        x, _, _ = self.process_encode_block(x)
        x = self.process_mid_block(x)
        x = self.pool_layer(x)
        x = self.out(x)
        
        return x
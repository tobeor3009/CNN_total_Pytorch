import torch
import numpy as np
from torch import nn
from .swin_layers import PatchEmbed, PatchMerging, PatchExpanding
from .swin_layers import BasicLayerV1, BasicLayerV2
from .swin_layers import trunc_normal_, to_2tuple
from ..layers import ConvBlock2D, AttentionPool1d
from ..layers import get_act
from einops.layers.torch import Rearrange
from torch.fft import fft2, ifft2
from swin_layers_diffusion import exists, default, extract


class SwinDiffusion(nn.Module):
    def __init__(self, img_size=512, patch_size=4, in_chans=3, out_chans=1, out_act="sigmoid",
                 embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_sizes=[8, 4, 4, 2], mlp_ratio=4., qkv_bias=True, ape=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, skip_connect=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0],
                 con_list=[], con_channel_list=[]
                 ):
        super().__init__()

        patch_size = int(patch_size)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.skip_connect = skip_connect

        time_dim = embed_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim,
                                      norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            pos_embed_shape = torch.zeros(1, num_patches, embed_dim)
            self.absolute_pos_embed = nn.Parameter(pos_embed_shape)
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        # build layers
        self.encode_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            BasicLayer = BasicLayerV1 if i_layer == 0 else BasicLayerV2
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_sizes[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(
                                   depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (
                i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                pretrained_window_size=pretrained_window_sizes[i_layer])
            self.encode_layers.append(layer)
        depth_level = self.num_layers - 1
        feature_hw = (patches_resolution[0] // (2 ** depth_level),
                      patches_resolution[1] // (2 ** depth_level))

        self.cat_linears = nn.ModuleList()
        self.decode_layers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1, -1, -1):
            target_dim = int(embed_dim * 2 ** i_layer)
            if i_layer > 0 and skip_connect:
                cat_linear = nn.Linear(target_dim * 2, target_dim,
                                        bias=False)
            else:
                cat_linear = nn.Identity()
            layer = BasicLayerV2(dim=target_dim,
                                    input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                    patches_resolution[1] // (2 ** i_layer)),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    window_size=window_sizes[i_layer],
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(
                                        depths[:i_layer + 1])],
                                    norm_layer=norm_layer,
                                    upsample=PatchExpanding if (
                                        i_layer > 0) else None,
                                    use_checkpoint=use_checkpoint,
                                    pretrained_window_size=pretrained_window_sizes[i_layer])
            self.cat_linears.append(cat_linear)
            self.decode_layers.append(layer)
        for bly in self.decode_layers:
            bly._init_respostnorm()
        self.seg_final_expanding = PatchExpanding(input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                                    patches_resolution[1] // (2 ** i_layer)),
                                                    dim=target_dim,
                                                    return_vector=False,
                                                    dim_scale=patch_size,
                                                    norm_layer=norm_layer
                                                    )
        self.seg_final_conv = nn.Conv2d(target_dim // 2, out_chans,
                                        kernel_size=1, padding=0)
        self.seg_final_act = get_act(out_act)

        self.apply(self._init_weights)
        for bly in self.encode_layers:
            bly._init_respostnorm()

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
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def encode_forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        skip_connect_list = []
        for idx, layer in enumerate(self.encode_layers):
            x = layer(x)
            if idx < len(self.encode_layers) - 1:
                skip_connect_list.insert(0, x)
        return x, skip_connect_list

    def decode_forward(self, x, skip_connect_list):
        for idx, (cat_linear, layer) in enumerate(zip(self.cat_linears,
                                                      self.decode_layers)):
            if idx < len(self.decode_layers) - 1 and self.skip_connect:
                skip_connect = skip_connect_list[idx]
                x = torch.cat([x, skip_connect], dim=-1)
                x = cat_linear(x)
            x = layer(x)
        x = self.seg_final_expanding(x)
        x = self.seg_final_conv(x)
        x = self.seg_final_act(x)
        return x


    def forward(self, x, time, cond=None, x_self_cond=None):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        skip_connect_list = []
        for idx, layer in enumerate(self.encode_layers):
            x = layer(x)
            if idx < len(self.encode_layers) - 1:
                skip_connect_list.insert(0, x)

        for idx, (cat_linear, layer) in enumerate(zip(self.cat_linears,
                                                      self.decode_layers)):
            if idx < len(self.decode_layers) - 1 and self.skip_connect:
                skip_connect = skip_connect_list[idx]
                x = torch.cat([x, skip_connect], dim=-1)
                x = cat_linear(x)
            x = layer(x)
        x = self.seg_final_expanding(x)
        x = self.seg_final_conv(x)
        x = self.seg_final_act(x)
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

    def print_tensor_info(self, tensor):
        print(tensor.min(), tensor.max(), torch.isnan(tensor).any())

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
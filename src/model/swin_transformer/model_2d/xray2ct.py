import torch
import numpy as np
from torch import nn
from ..model_3d.swin_layers import PatchExpanding as PatchExpanding3D
from ..model_3d.swin_layers import BasicLayerV1 as BasicLayerV1_3D
from ..model_3d.swin_layers import BasicLayerV2 as BasicLayerV2_3D
from .swin_layers import PatchEmbed, PatchMerging, PatchExpanding
from .swin_layers import PatchExtract, PatchEmbedding
from .swin_layers import BasicLayerV1, BasicLayerV2
from .swin_layers import trunc_normal_, to_2tuple
from ..layers import ConvBlock2D, AttentionPool1d
from ..layers import get_act
import math


class SwinXray2CT(nn.Module):
    def __init__(self, img_size=256, patch_size=4, in_chans=4,
                 embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_sizes=[8, 4, 4, 2], mlp_ratio=4., qkv_bias=True, ape=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 last_act="sigmoid",
                 norm_layer=nn.LayerNorm, patch_norm=True, skip_connect=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0]):
        super().__init__()

        patch_size = int(patch_size)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.skip_connect = skip_connect
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

        self.layer_2d_3d = PatchExpanding2D_3D(feature_hw,
                                               int(embed_dim * 2 ** i_layer))
        self.cat_linears = nn.ModuleList()
        self.decode_layers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1, -1, -1):
            target_dim = int(embed_dim * 2 ** i_layer)
            if i_layer > 0 and skip_connect:
                cat_linear = nn.Linear(target_dim * 2, target_dim,
                                       bias=False)
            else:
                cat_linear = nn.Identity()
            layer = BasicLayerV2_3D(dim=target_dim,
                                    input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                      patches_resolution[0] // (
                                                          2 ** i_layer),
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
                                    upsample=PatchExpanding3D if (
                                        i_layer > 0) else None,
                                    use_checkpoint=use_checkpoint,
                                    pretrained_window_size=pretrained_window_sizes[i_layer])
            self.cat_linears.append(cat_linear)
            self.decode_layers.append(layer)
        for bly in self.decode_layers:
            bly._init_respostnorm()

        self.seg_final_expanding_list = nn.ModuleList([])
        expanding_num = int(math.log2(patch_size))
        for idx in range(expanding_num):
            seg_final_expanding = PatchExpanding3D(input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                                     patches_resolution[0] // (
                2 ** i_layer),
                patches_resolution[1] // (2 ** i_layer)),
                dim=target_dim // (2 ** idx),
                return_vector=False if idx == (expanding_num - 1) else True,
                dim_scale=patch_size,
                norm_layer=norm_layer)
            self.seg_final_expanding_list.append(seg_final_expanding)
        self.seg_final_conv = nn.Conv3d(target_dim // (2 ** expanding_num), 1,
                                        kernel_size=1, padding=0)
        self.seg_final_act = get_act(last_act)
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
        for seg_final_expanding in self.seg_final_expanding_list:
            x = seg_final_expanding(x)
        x = self.seg_final_conv(x)
        x = self.seg_final_act(x)
        return x

    def forward(self, x):
        output = []
        x, skip_connect_list = self.encode_forward(x)
        x = self.layer_2d_3d(x)
        seg_output = self.decode_forward(x, skip_connect_list)
        return seg_output

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


class PatchExpanding2D_3D(nn.Module):
    def __init__(self, feature_hw, embed_dim):
        super().__init__()
        power = int(math.log(feature_hw[0], 2))
        power_4 = power // 3
        power_2 = power - power_4 * 2

        self.expand_2d_list = nn.ModuleList([])
        self.expand_3d_list = nn.ModuleList([])

        for idx_2d in range(power_2):
            h_ratio = 2 ** (idx_2d // 2)
            w_ratio = 2 ** (idx_2d // 2 + idx_2d % 2)
            expand_2d = PatchExpanding(input_resolution=(feature_hw[0] * h_ratio,
                                                         feature_hw[1] * w_ratio),
                                       dim=embed_dim,
                                       return_vector=True)
            self.expand_2d_list.append(expand_2d)

        up_ratio_2d = 2 ** (idx_2d + 1)
        for idx_3d in range(power_4):
            up_ratio_3d = 4 ** idx_3d
            upsample_resolution = (up_ratio_2d * up_ratio_3d, *feature_hw)
            expand_3d = PatchExpanding3D(input_resolution=upsample_resolution, dim=embed_dim,
                                         return_vector=True)
            self.expand_3d_list.append(expand_3d)

    def forward(self, x):
        for expand_2d in self.expand_2d_list:
            x = self.block_process(x, expand_2d)
        for expand_3d in self.expand_3d_list:
            x = self.block_process(x, expand_3d)
        return x

    def block_process(self, x, expand_block):
        x = expand_block(x)
        B, N, C = x.shape
        x = x.view(B, 2, N // 2, C).permute(0, 2, 1, 3)
        x = x.reshape(B, N // 2, C * 2)
        return x

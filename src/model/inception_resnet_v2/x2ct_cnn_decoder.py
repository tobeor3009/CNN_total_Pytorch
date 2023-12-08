import torch
import math
from torch import nn
import numpy as np
from timm.models.layers import trunc_normal_
from .common_module.base_model import InceptionResNetV2_2D, get_skip_connect_channel_list
from .common_module.layers import get_act, get_norm
from .common_module.transformer_layers import PositionalEncoding
from .common_module.layers import ConvBlock1D, ConvBlock2D, ConvBlock3D, Output3D
from .common_module.layers_highway import MultiDecoder3D, HighwayOutput3D
from ..swin_transformer.model_2d.swin_layers import PatchEmbed as PatchEmbed2D
from ..swin_transformer.model_2d.swin_layers import PatchExpandingConcat as PatchExpanding2D
from ..swin_transformer.model_2d.swin_layers import BasicLayerV2 as BasicLayerV2_2D
from ..swin_transformer.model_3d.swin_layers import PatchExpandingConcat as PatchExpanding3D
from ..swin_transformer.model_3d.swin_layers import BasicLayerV2 as BasicLayerV2_3D


USE_INPLACE = True


class InceptionResNetV2_X2CT(nn.Module):
    def __init__(self, input_shape, seg_channels=1,
                 conv_norm="instance", conv_act="leakyrelu",
                 trans_norm=nn.LayerNorm, trans_act="relu6",
                 cnn_block_size=16, decode_init_channel=None,
                 patch_size=4, depths=[2, 2, 2, 2, 2], num_heads=[4, 2, 2, 1, 1],
                 window_sizes=[2, 2, 2, 4, 4], mlp_ratio=4.0,
                 seg_act="sigmoid",
                 ):
        super().__init__()

        if decode_init_channel is None:
            decode_init_channel = cnn_block_size * 32
        skip_connect_channel_list = get_skip_connect_channel_list(
            cnn_block_size)
        input_shape = np.array(input_shape)
        n_input_channels, init_h, init_w = input_shape
        feature_hw = (init_h // (2 ** 5),
                      init_w // (2 ** 5))
        feature_h, feature_w = feature_hw
        feature_channel_num = cnn_block_size * 96

        self.feature_shape = np.array([feature_channel_num,
                                       input_shape[1] // 32,
                                       input_shape[2] // 32])

        self.base_model = InceptionResNetV2_2D(n_input_channels=n_input_channels,
                                               block_size=cnn_block_size,
                                               padding="same", norm=conv_norm, act=conv_act,
                                               include_cbam=False, include_context=False,
                                               include_skip_connection_tensor=True)
        self.decode_init_conv = ConvBlock2D(in_channels=feature_channel_num,
                                            out_channels=decode_init_channel,
                                            kernel_size=1, norm=conv_norm, act=conv_act)
        self.decode_init_trans = PatchExpanding2D_3D(feature_hw=feature_hw,
                                                     embed_dim=decode_init_channel,
                                                     patch_size=patch_size, depth=depths[0],
                                                     num_head=num_heads[0], window_size=window_sizes[0],
                                                     mlp_ratio=mlp_ratio, norm_layer=trans_norm)
        for decode_i in range(0, 5):
            down_ratio = 2 ** (5 - decode_i)
            channel_down_ratio = 2 ** decode_i
            resolution_3d = (init_h // down_ratio,
                             init_h // down_ratio,
                             init_w // down_ratio)
            decode_in_channels = int(decode_init_channel //
                                     channel_down_ratio)
            decode_out_channels = int(decode_in_channels // 2)
            skip_conv_channel = decode_in_channels
            if decode_i > 0:
                skip_conv_channel += decode_in_channels // 2

            skip_hw = np.array(feature_hw) * (channel_down_ratio)
            skip_channel = skip_connect_channel_list[4 - decode_i]
            skip_2d_conv = ConvBlock2D(in_channels=skip_channel,
                                       out_channels=decode_in_channels,
                                       kernel_size=1, norm=conv_norm, act=conv_act)
            skip_2d_3d = PatchExpanding2D_3D(feature_hw=skip_hw,
                                             embed_dim=decode_in_channels,
                                             patch_size=patch_size, depth=depths[decode_i],
                                             num_head=num_heads[decode_i],
                                             window_size=window_sizes[decode_i],
                                             mlp_ratio=mlp_ratio, norm_layer=trans_norm)
            skip_conv = ConvBlock3D(in_channels=skip_conv_channel,
                                    out_channels=decode_in_channels,
                                    kernel_size=1)
            setattr(self, f"decode_skip_2d_conv_{decode_i}", skip_2d_conv)
            setattr(self, f"decode_skip_2d_3d_{decode_i}", skip_2d_3d)
            setattr(self, f"decode_skip_conv_{decode_i}", skip_conv)

            decode_conv = ConvBlock3D(in_channels=decode_in_channels,
                                      out_channels=decode_out_channels, kernel_size=3)
            decode_kernel_size = 2
            decode_up = MultiDecoder3D(input_zhw=resolution_3d,
                                       in_channels=decode_out_channels,
                                       out_channels=decode_out_channels,
                                       kernel_size=decode_kernel_size,
                                       use_highway=False,
                                       use_pixelshuffle_only=False)
            setattr(self, f"decode_conv_{decode_i}", decode_conv)
            setattr(self, f"decode_up_{decode_i}", decode_up)
        resolution_3d = np.array(resolution_3d) * 2
        decode_out_channels = decode_in_channels // 2
        self.seg_final_conv = HighwayOutput3D(in_channels=decode_out_channels,
                                              out_channels=seg_channels,
                                              act=seg_act, use_highway=False)

    def forward(self, input_tensor):
        encode_feature = self.base_model(input_tensor)
        decoded = encode_feature
        decoded = self.decode_init_conv(decoded)
        decoded = self.decode_init_trans(decoded)
        for decode_i in range(0, 5):
            decode_skip_2d_conv = getattr(self,
                                          f"decode_skip_2d_conv_{decode_i}")
            decode_skip_2d_3d = getattr(self,
                                        f"decode_skip_2d_3d_{decode_i}")
            decode_skip_conv = getattr(self,
                                       f"decode_skip_conv_{decode_i}")
            skip_connect_tensor = getattr(self.base_model,
                                          f"skip_connect_tensor_{4 - decode_i}")

            skip_connect_tensor = decode_skip_2d_conv(skip_connect_tensor)
            skip_connect_tensor = decode_skip_2d_3d(skip_connect_tensor)
            decoded = torch.cat([decoded,
                                skip_connect_tensor], dim=1)
            decoded = decode_skip_conv(decoded)

            decode_conv = getattr(self, f"decode_conv_{decode_i}")
            decode_up = getattr(self, f"decode_up_{decode_i}")
            decoded = decode_conv(decoded)
            decoded = decode_up(decoded)
        seg_output = self.seg_final_conv(decoded)
        return seg_output


class PatchExpanding2D_3D(nn.Module):
    def __init__(self, feature_hw, embed_dim, patch_size,
                 depth, num_head, window_size, mlp_ratio, norm_layer):
        super().__init__()
        power = int(math.log(feature_hw[0] // patch_size, 2))
        power_4 = power // 3
        power_2 = power - power_4 * 2

        self.patch_embed_2d = PatchEmbed2D(img_size=feature_hw[0],
                                           patch_size=patch_size,
                                           in_chans=embed_dim,
                                           embed_dim=embed_dim,
                                           norm_layer=norm_layer)

        num_patches = self.patch_embed_2d.num_patches
        patches_resolution = self.patch_embed_2d.patches_resolution

        pos_embed_shape = torch.zeros(
            1, num_patches * patches_resolution[0], embed_dim)
        self.absolute_pos_embed = nn.Parameter(pos_embed_shape)
        trunc_normal_(self.absolute_pos_embed, std=.02)

        self.expand_2d_list = nn.ModuleList([])
        self.expand_3d_list = nn.ModuleList([])
        dpr = [x.item() for x in torch.linspace(0, 0.05,
                                                (power_2 + power_4) * depth)]

        for idx_2d in range(power_2):
            h_ratio = 2 ** (idx_2d // 2)
            w_ratio = 2 ** (idx_2d // 2 + idx_2d % 2)
            resolution_2d = (patches_resolution[0] * h_ratio,
                             patches_resolution[1] * w_ratio)
            expand_2d = BasicLayerV2_2D(dim=embed_dim,
                                        input_resolution=resolution_2d,
                                        depth=depth,
                                        num_heads=num_head,
                                        window_size=window_size,
                                        mlp_ratio=mlp_ratio,
                                        qkv_bias=True,
                                        drop=0.0, attn_drop=0.0,
                                        drop_path=dpr[idx_2d * depth:
                                                      (idx_2d + 1) * depth],
                                        norm_layer=norm_layer,
                                        upsample=PatchExpanding2D)
            self.expand_2d_list.append(expand_2d)

        up_ratio_2d = 2 ** (idx_2d + 1)
        for idx_3d in range(power_4):
            up_ratio_3d = 4 ** idx_3d
            resolution_3d = (up_ratio_2d * up_ratio_3d,
                             *patches_resolution)
            expand_3d = BasicLayerV2_3D(dim=embed_dim,
                                        input_resolution=resolution_3d,
                                        depth=depth,
                                        num_heads=num_head,
                                        window_size=window_size,
                                        mlp_ratio=mlp_ratio,
                                        qkv_bias=True,
                                        drop=0.0, attn_drop=0.0,
                                        drop_path=dpr[(idx_2d + idx_3d) * depth:
                                                      (idx_2d + idx_3d + 1) * depth],
                                        norm_layer=norm_layer,
                                        upsample=PatchExpanding3D)
            self.expand_3d_list.append(expand_3d)
        resolution_3d = [feature_hw[0] // patch_size for _ in range(3)]
        self.final_expand = PatchExpanding3D(input_resolution=resolution_3d,
                                             dim=embed_dim,
                                             return_vector=False,
                                             dim_scale=patch_size,
                                             norm_layer=norm_layer)

    def forward(self, x):
        x = self.patch_embed_2d(x)
        for expand_2d in self.expand_2d_list:
            x = self.block_process(x, expand_2d)
        for expand_3d in self.expand_3d_list:
            x = self.block_process(x, expand_3d)
        x = x + self.absolute_pos_embed
        x = self.final_expand(x)
        return x

    def block_process(self, x, expand_block):
        x = expand_block(x)
        B, N, C = x.shape
        x = x.view(B, 2, N // 2, C).permute(0, 2, 1, 3)
        x = x.reshape(B, N // 2, C * 2)
        return x

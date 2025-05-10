import torch
import math
from torch import nn
import numpy as np
from timm.models.layers import trunc_normal_
from .resnet_2d import resnet
from ..common_module.layers import get_act, get_norm
from ..common_module.transformer_layers import PositionalEncoding
from ..common_module.layers import ConvBlock2D, ConvBlock3D, Output3D, SkipUpSample3D
from ..common_module.layers_highway import MultiDecoder3D, HighwayOutput3D
from ...swin_transformer.model_2d.swin_layers import PatchEmbed as PatchEmbed2D
from ...swin_transformer.model_2d.swin_layers import PatchExpanding as PatchExpanding2D
from ...swin_transformer.model_3d.swin_layers import PatchExpanding as PatchExpanding3D

USE_INPLACE = True


class ResNetX2CT(nn.Module):
    def __init__(self, input_shape, seg_channels=1,
                 patch_size=4,
                 block_size=64, block_depth_list=[3, 4, 6, 3], decode_init_channels=None,
                 seg_act="sigmoid",
                 ):
        super().__init__()

        if decode_init_channels is None:
            decode_init_channel = block_size * 16
        skip_connect_channel_list = [block_size,
                                     block_size * 4,
                                     block_size * 8,
                                     block_size * 16]
        input_shape = np.array(input_shape)
        n_input_channels, init_h, init_w = input_shape
        feature_h, feature_w = (init_h // (2 ** 5),
                                init_w // (2 ** 5),)

        feature_channel_num = block_size * 32

        self.feature_shape = np.array([feature_channel_num,
                                       input_shape[1] // 32,
                                       input_shape[2] // 32])

        self.base_model = resnet(in_channel=n_input_channels,
                                 block_size=block_size,
                                 block_depth_list=block_depth_list)
        self.decode_init_conv = SkipUpSample3D(in_channels=feature_channel_num,
                                               out_channels=decode_init_channel,
                                               cbam=cbam)
        for decode_i in range(0, 5):
            z, h, w = (init_h // (2 ** (5 - decode_i)),
                       init_h // (2 ** (5 - decode_i)),
                       init_w // (2 ** (5 - decode_i)))
            decode_in_channels = int(decode_init_channel //
                                     (2 ** decode_i))
            if decode_i > 0:
                skip_2d_3d = SkipUpSample3D(in_channels=skip_connect_channel_list[-decode_i],
                                            out_channels=decode_in_channels,
                                            cbam=cbam)
                skip_conv = nn.Conv3d(in_channels=decode_in_channels * 2,
                                      out_channels=decode_in_channels, kernel_size=1)
                setattr(self, f"decode_skip_2d_3d_{decode_i}", skip_2d_3d)
                setattr(self, f"decode_skip_conv_{decode_i}", skip_conv)

            decode_out_channels = decode_in_channels // 2
            decode_conv = ConvBlock3D(in_channels=decode_in_channels,
                                      out_channels=decode_out_channels,
                                      kernel_size=3)
            decode_kernel_size = (2, 2, 2)
            decode_up = MultiDecoder3D(input_zhw=(z, h, w),
                                       in_channels=decode_out_channels,
                                       out_channels=decode_out_channels,
                                       kernel_size=decode_kernel_size,
                                       use_highway=False,
                                       use_pixelshuffle_only=use_pixelshuffle_only)
            setattr(self, f"decode_conv_{decode_i}", decode_conv)
            setattr(self, f"decode_up_{decode_i}", decode_up)
        self.seg_output_conv = Output3D(in_channels=decode_out_channels,
                                        out_channels=seg_channels,
                                        act=seg_act)

    def forward(self, input_tensor):
        encode_feature, skip_connect_list = self.base_model(input_tensor)
        decoded = encode_feature
        decoded = self.decode_init_conv(decoded)
        for decode_i in range(0, 5):
            if decode_i > 0:
                decoded_skip_2d_3d = getattr(self,
                                             f"decode_skip_2d_3d_{decode_i}")
                decoded_skip_conv = getattr(self,
                                            f"decode_skip_conv_{decode_i}")
                skip_connect_tensor = skip_connect_list[-decode_i]
                skip_connect_tensor = decoded_skip_2d_3d(skip_connect_tensor)
                decoded = torch.cat([decoded,
                                    skip_connect_tensor], dim=1)
                decoded = decoded_skip_conv(decoded)
            decode_conv = getattr(self, f"decode_conv_{decode_i}")
            decode_up = getattr(self, f"decode_up_{decode_i}")
            decoded = decode_conv(decoded)
            decoded = decode_up(decoded)
        seg_output = self.seg_output_conv(decoded)
        return seg_output


class PatchExpanding2D_3D(nn.Module):
    def __init__(self, feature_hw, embed_dim, patch_size):
        super().__init__()
        power = int(math.log(feature_hw[0], 2))
        power_4 = power // 3
        power_2 = power - power_4 * 2

        self.patch_extract_2d = PatchEmbed2D(img_size=feature_hw[0],
                                             stride_size=patch_size,
                                             in_chans=embed_dim,
                                             embed_dim=embed_dim,

                                             )

        self.expand_2d_list = nn.ModuleList([])
        self.expand_3d_list = nn.ModuleList([])

        for idx_2d in range(power_2):
            h_ratio = 2 ** (idx_2d // 2)
            w_ratio = 2 ** (idx_2d // 2 + idx_2d % 2)
            expand_2d = PatchExpanding2D(input_resolution=(feature_hw[0] * h_ratio,
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

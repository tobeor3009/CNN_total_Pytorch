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
USE_INPLACE = True


class ResNetX2CT(nn.Module):
    def __init__(self, input_shape, seg_channels=1, cbam=False,
                 block_size=64, block_depth_list=[3, 4, 6, 3], decode_init_channels=None,
                 seg_act="sigmoid", use_seg_pixelshuffle_only=True, use_seg_simpleoutput=False
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
            if decode_i == 0:
                use_pixelshuffle_only = True
            else:
                use_pixelshuffle_only = use_seg_pixelshuffle_only
            decode_up = MultiDecoder3D(input_zhw=(z, h, w),
                                       in_channels=decode_out_channels,
                                       out_channels=decode_out_channels,
                                       kernel_size=decode_kernel_size,
                                       use_highway=False,
                                       use_pixelshuffle_only=use_pixelshuffle_only)
            setattr(self, f"decode_conv_{decode_i}", decode_conv)
            setattr(self, f"decode_up_{decode_i}", decode_up)
        if use_seg_simpleoutput:
            self.seg_output_conv = Output3D(in_channels=decode_out_channels,
                                            out_channels=seg_channels,
                                            act=seg_act)
        else:
            self.seg_output_conv = HighwayOutput3D(in_channels=decode_out_channels,
                                                   out_channels=seg_channels,
                                                   act=seg_act, use_highway=False)

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

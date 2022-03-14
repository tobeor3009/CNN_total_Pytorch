import torch
import numpy as np
from torch import nn
from .transformer_layers import PositionalEncoding
from .base_model import InceptionResNetV2_2D, InceptionResNetV2_3D, get_skip_connect_channel_list
from .layers import ConvBlock3D, SkipUpSample3D, Decoder3D, HighwayOutput3D
from reformer_pytorch import Reformer
from einops import rearrange


class APLATX2CTGenerator(nn.Module):
    def __init__(self, xray_shape, ct_series_shape, block_size=16,
                 decode_init_channel=768,
                 include_cbam=True, include_context=False,
                 dropout_proba=0.1
                 ):
        super().__init__()
        n_input_channels = 1
        n_output_channels = 1
        skip_connect = True
        feature_2d_channel_num = block_size * 96
        feature_3d_channel_num = feature_2d_channel_num // 16
        self.feature_shape = np.array([feature_2d_channel_num,
                                       xray_shape[1] // 32,
                                       xray_shape[2] // 32])
        skip_connect_channel_list = get_skip_connect_channel_list(block_size)

        self.ct_z_dim = 16
        if ct_series_shape == (256, 256, 256):
            self.decode_start_index = 1
        elif ct_series_shape == (128, 128, 128):
            self.decode_start_index = 2
        elif ct_series_shape == (64, 64, 64):
            self.decode_start_index = 3
        else:
            NotImplementedError(
                "ct_series_shape is implemented only 64, 128, 256 intercubic shape")

        self.ap_model = InceptionResNetV2_2D(n_input_channels=n_input_channels, block_size=block_size,
                                             padding="same", include_cbam=include_cbam, include_context=include_context,
                                             include_skip_connection_tensor=skip_connect)
        self.lat_model = InceptionResNetV2_2D(n_input_channels=n_input_channels, block_size=block_size,
                                              padding="same", include_cbam=include_cbam, include_context=include_context,
                                              include_skip_connection_tensor=skip_connect)

        self.ap_positional_encoding = PositionalEncoding(d_model=self.feature_shape[1] * self.feature_shape[2],
                                                         dropout=dropout_proba)
        self.lat_positional_encoding = PositionalEncoding(d_model=self.feature_shape[1] * self.feature_shape[2],
                                                          dropout=dropout_proba)
        self.ap_encoder = Reformer(
            dim=feature_2d_channel_num,
            depth=6,
            heads=8,
            lsh_dropout=0.1,
            causal=True
        )
        self.lat_encoder = Reformer(
            dim=feature_2d_channel_num,
            depth=6,
            heads=8,
            lsh_dropout=0.1,
            causal=True
        )

        ap_skip_connect_conv_list = []
        ap_decode_conv_list = []
        ap_decode_up_list = []
        lat_skip_connect_conv_list = []
        lat_decode_conv_list = []
        lat_decode_up_list = []
        concat_decode_conv_list = []
        concat_decode_up_list = []

        for index, decode_i in enumerate(range(self.decode_start_index, 5)):
            if index > 0:
                decode_in_channels = decode_init_channel // (
                    2 ** (index - 1))
            else:
                decode_in_channels = feature_3d_channel_num
            skip_connect_channel = skip_connect_channel_list[4 - index]
            decode_out_channels = decode_init_channel // (2 ** index)
            ap_skip_connect_conv = SkipUpSample3D(in_channels=skip_connect_channel,
                                                  out_channels=decode_in_channels)
            lat_skip_connect_conv = SkipUpSample3D(in_channels=skip_connect_channel,
                                                   out_channels=decode_in_channels)
            ap_skip_connect_conv_list.append(ap_skip_connect_conv)
            lat_skip_connect_conv_list.append(lat_skip_connect_conv)

            ap_decode_conv_1 = ConvBlock3D(in_channels=decode_in_channels * 2,
                                           out_channels=decode_out_channels, kernel_size=3)
            ap_decode_conv_2 = ConvBlock3D(in_channels=decode_out_channels,
                                           out_channels=decode_out_channels, kernel_size=3)
            lat_decode_conv_1 = ConvBlock3D(in_channels=decode_in_channels * 2,
                                            out_channels=decode_out_channels, kernel_size=3)
            lat_decode_conv_2 = ConvBlock3D(in_channels=decode_out_channels,
                                            out_channels=decode_out_channels, kernel_size=3)
            concat_decode_conv_1 = ConvBlock3D(in_channels=decode_in_channels,
                                               out_channels=decode_out_channels, kernel_size=3)
            concat_decode_conv_2 = ConvBlock3D(in_channels=decode_out_channels,
                                               out_channels=decode_out_channels, kernel_size=3)
            ap_decode_conv_list.append(ap_decode_conv_1)
            ap_decode_conv_list.append(ap_decode_conv_2)
            lat_decode_conv_list.append(lat_decode_conv_1)
            lat_decode_conv_list.append(lat_decode_conv_2)
            concat_decode_conv_list.append(concat_decode_conv_1)
            concat_decode_conv_list.append(concat_decode_conv_2)

            if decode_i < 4:
                ap_decode_up = Decoder3D(in_channels=decode_out_channels,
                                         out_channels=decode_out_channels)
                lat_decode_up = Decoder3D(in_channels=decode_out_channels,
                                          out_channels=decode_out_channels)
                ap_decode_up_list.append(ap_decode_up)
                lat_decode_up_list.append(lat_decode_up)

            concat_decode_up = Decoder3D(in_channels=decode_out_channels,
                                         out_channels=decode_out_channels)
            concat_decode_up_list.append(concat_decode_up)
            self.ct_z_dim *= 2

        self.ct_z_dim = 16
        self.ap_skip_connect_conv_list = nn.ModuleList(
            ap_skip_connect_conv_list)
        self.ap_decode_conv_list = nn.ModuleList(ap_decode_conv_list)
        self.ap_decode_up_list = nn.ModuleList(ap_decode_up_list)
        self.lat_skip_connect_conv_list = nn.ModuleList(
            lat_skip_connect_conv_list)
        self.lat_decode_conv_list = nn.ModuleList(lat_decode_conv_list)
        self.lat_decode_up_list = nn.ModuleList(lat_decode_up_list)
        self.concat_decode_conv_list = nn.ModuleList(concat_decode_conv_list)
        self.concat_decode_up_list = nn.ModuleList(concat_decode_up_list)

        self.output_conv = HighwayOutput3D(in_channels=decode_out_channels,
                                           out_channels=n_output_channels)

    def forward(self, xray_tensor):
        ap_tensor = xray_tensor[:, 0:1, :, :]
        lat_tensor = xray_tensor[:, 1:2, :, :]

        ap_feature = self.ap_model(ap_tensor)
        lat_feature = self.lat_model(lat_tensor)

        ap_decoded = rearrange(ap_feature, 'b c h w -> b (h w) c')
        ap_decoded = self.ap_positional_encoding(ap_decoded)
        ap_decoded = self.ap_encoder(ap_decoded)
        ap_decoded = rearrange(ap_decoded, 'b (h w) (c z) -> b c z h w',
                               h=self.feature_shape[1],
                               w=self.feature_shape[2],
                               z=self.ct_z_dim)
        lat_decoded = rearrange(lat_feature, 'b c h w -> b (h w) c')
        lat_decoded = self.lat_positional_encoding(lat_decoded)
        lat_decoded = self.lat_encoder(lat_decoded)
        lat_decoded = rearrange(lat_decoded, 'b (h w) (c z) -> b c z h w',
                                h=self.feature_shape[1],
                                w=self.feature_shape[2],
                                z=self.ct_z_dim)
        concat_decoded = (ap_decoded + lat_decoded) / 2

        for index, decode_i in enumerate(range(self.decode_start_index, 5)):

            ap_skip_connect_conv = self.ap_skip_connect_conv_list[index]
            lat_skip_connect_conv = self.lat_skip_connect_conv_list[index]

            ap_decode_conv_1 = self.ap_decode_conv_list[2 * index]
            ap_decode_conv_2 = self.ap_decode_conv_list[2 * index + 1]
            lat_decode_conv_1 = self.lat_decode_conv_list[2 * index]
            lat_decode_conv_2 = self.lat_decode_conv_list[2 * index + 1]
            concat_decode_conv_1 = self.concat_decode_conv_list[2 * index]
            concat_decode_conv_2 = self.concat_decode_conv_list[2 * index + 1]

            ap_skip_connect = self.ap_model.skip_connect_tensor_list[4 - index]
            ap_skip_connect = ap_skip_connect_conv(ap_skip_connect)
            lat_skip_connect = self.lat_model.skip_connect_tensor_list[4 - index]
            lat_skip_connect = lat_skip_connect_conv(lat_skip_connect)
            ap_decoded = torch.cat([ap_decoded, ap_skip_connect], dim=1)
            lat_decoded = torch.cat([lat_decoded, lat_skip_connect], dim=1)

            ap_decoded = ap_decode_conv_1(ap_decoded)
            ap_decoded = ap_decode_conv_2(ap_decoded)
            lat_decoded = lat_decode_conv_1(lat_decoded)
            lat_decoded = lat_decode_conv_2(lat_decoded)
            concat_decoded = concat_decode_conv_1(concat_decoded)
            concat_decoded = (concat_decoded + ap_decoded + lat_decoded) / 3
            concat_decoded = concat_decode_conv_2(concat_decoded)

            if decode_i < 4:
                ap_decode_up = self.ap_decode_up_list[index]
                lat_decode_up = self.lat_decode_up_list[index]
                ap_decoded = ap_decode_up(ap_decoded)
                lat_decoded = lat_decode_up(lat_decoded)
            concat_decode_up = self.concat_decode_up_list[index]
            concat_decoded = concat_decode_up(concat_decoded)

        output_ct = self.output_conv(concat_decoded)
        return output_ct


class X2CTDiscriminator(nn.Module):
    def __init__(self, ct_series_shape, block_size=16,
                 dropout_proba=0.1, include_context=False,
                 ):
        super().__init__()
        n_input_channels = 2
        ct_series_shape = np.array(ct_series_shape)
        feature_shape = ct_series_shape // 32
        self.block_size = block_size
        self.include_context = include_context
        self.base_model = InceptionResNetV2_3D(n_input_channels=n_input_channels, block_size=block_size,
                                               padding="same", z_channel_preserve=False,
                                               include_context=include_context)
        # (block_size * 96) / block_size
        feature_channel_num = 96
        final_feature_num = np.prod(feature_shape) * block_size
        # current feature shape: [1536 8 8 8]
        # current feature shape: [1536 8 * 8 * 8]
        # current feature shape: [512 1536]
        self.positional_encoding = PositionalEncoding(d_model=final_feature_num,
                                                      dropout=dropout_proba)
        self.transformer_encoder = Reformer(
            dim=feature_channel_num,
            depth=6,
            heads=8,
            bucket_size=final_feature_num // 8,
            lsh_dropout=0.1,
            causal=True
        )

    def forward(self, x):
        x = self.base_model(x)
        x = rearrange(x, 'b (c d) z h w -> b (z h w d) c',
                      d=self.block_size)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        output = torch.sigmoid(x)
        return output

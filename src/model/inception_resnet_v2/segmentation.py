import torch
from torch import nn
import numpy as np
from .base_model import InceptionResNetV2_2D, get_skip_connect_channel_list
from .transformer_layers import PositionalEncoding
from reformer_pytorch import Reformer
from .layers import ConvBlock2D, Decoder2D, HighwayOutput2D
from einops import rearrange


class SegInceptionResNetV22D(nn.Module):
    def __init__(self, input_shape, n_output_channels, block_size=16,
                 include_cbam=True, include_context=False, decode_init_channel=768,
                 skip_connect=True, version="reformer", dropout_proba=0.1
                 ):
        super().__init__()
        input_shape = np.array(input_shape)
        n_input_channels = input_shape[0]
        feature_channel_num = block_size * 96
        self.feature_shape = np.array([feature_channel_num,
                                       input_shape[1] // 32,
                                       input_shape[2] // 32])
        self.skip_connect = skip_connect
        skip_connect_channel_list = get_skip_connect_channel_list(block_size)

        self.base_model = InceptionResNetV2_2D(n_input_channels=n_input_channels, block_size=block_size,
                                               padding="same", include_cbam=include_cbam, include_context=include_context,
                                               include_skip_connection_tensor=skip_connect)
        if version == "reformer":
            self.positional_encoding = PositionalEncoding(d_model=self.feature_shape[1] * self.feature_shape[2],
                                                          dropout=dropout_proba)
            self.encoder = Reformer(
                dim=feature_channel_num,
                depth=6,
                heads=8,
                lsh_dropout=0.1,
                causal=True
            )
        elif version == "normal":
            pass
        else:
            raise Exception("not Supported version")

        for decode_i in range(0, 5):
            decode_in_channels = decode_init_channel // (
                2 ** (decode_i - 1)) if decode_i > 0 else feature_channel_num
            if skip_connect:
                decode_in_channels += skip_connect_channel_list[4 - decode_i]
            decode_out_channels = decode_init_channel // (2 ** decode_i)
            decode_conv = ConvBlock2D(in_channels=decode_in_channels,
                                      out_channels=decode_out_channels, kernel_size=3)
            decode_up = Decoder2D(in_channels=decode_out_channels,
                                  out_channels=decode_out_channels, kernel_size=2)
            setattr(self, f"decode_conv_{decode_i}", decode_conv)
            setattr(self, f"decode_up_{decode_i}", decode_up)

        self.output_conv = HighwayOutput2D(in_channels=decode_out_channels,
                                           out_channels=n_output_channels)

    def forward(self, input_tensor):
        conv_feature = self.base_model(input_tensor)

        decoded = rearrange(conv_feature, 'b c h w -> b (h w) c')
        decoded = self.positional_encoding(decoded)
        decoded = self.encoder(decoded)
        decoded = rearrange(decoded, 'b (h w) c -> b c h w',
                            h=self.feature_shape[1],
                            w=self.feature_shape[2])

        for decode_i in range(0, 5):

            if self.skip_connect:
                skip_connect_tensor = getattr(self.base_model, f"skip_connect_tensor_{4 - decode_i}")
                decoded = torch.cat([decoded, skip_connect_tensor], axis=1)

            decode_conv = getattr(self, f"decode_conv_{decode_i}")
            decode_up = getattr(self, f"decode_up_{decode_i}")
            decoded = decode_conv(decoded)
            decoded = decode_up(decoded)
            
        output = self.output_conv(decoded)
        return output

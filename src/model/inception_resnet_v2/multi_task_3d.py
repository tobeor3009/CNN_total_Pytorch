import torch
import math
from torch import nn
import numpy as np
from .layers import get_act
from .base_model import InceptionResNetV2_3D, get_skip_connect_channel_list
from .transformer_layers import PositionalEncoding
from .layers import space_to_depth_3d
from .layers import ConvBlock3D, Decoder3D, HighwayOutput3D
from torch.nn import TransformerEncoder, TransformerEncoderLayer
USE_INPLACE = True


class InceptionResNetV2MultiTask3D(nn.Module):
    def __init__(self, input_shape, class_channel, seg_channels, block_size=16,
                 z_channel_preserve=False, include_context=False, decode_init_channel=768,
                 skip_connect=True, dropout_proba=0.05, class_act="softmax", seg_act="sigmoid",
                 get_seg=True, get_class=True, use_class_head_simple=True
                 ):
        super().__init__()

        self.get_seg = get_seg
        self.get_class = get_class

        input_shape = np.array(input_shape)
        n_input_channels = input_shape[0]
        feature_channel_num = block_size * 96
        self.feature_shape = np.array([feature_channel_num,
                                       input_shape[1] // 32,
                                       input_shape[2] // 32,
                                       input_shape[3] // 32])
        self.skip_connect = skip_connect

        skip_connect_channel_list = get_skip_connect_channel_list(block_size)

        self.base_model = InceptionResNetV2_3D(n_input_channels=n_input_channels, block_size=block_size,
                                               padding="same", z_channel_preserve=z_channel_preserve,
                                               include_context=include_context,
                                               include_skip_connection_tensor=skip_connect)
        if self.get_seg:
            for decode_i in range(0, 5):
                decode_in_channels = decode_init_channel // (
                    2 ** (decode_i - 1)) if decode_i > 0 else feature_channel_num
                if skip_connect:
                    decode_in_channels += skip_connect_channel_list[4 - decode_i]
                decode_out_channels = decode_init_channel // (2 ** decode_i)
                decode_conv = ConvBlock3D(in_channels=decode_in_channels,
                                          out_channels=decode_out_channels, kernel_size=3)
                decode_kernel_size = (1, 2, 2) if z_channel_preserve else 2
                use_highway = True if decode_i >= 4 else False
                decode_up = Decoder3D(in_channels=decode_out_channels,
                                      out_channels=decode_out_channels,
                                      kernel_size=decode_kernel_size,
                                      use_highway=use_highway)
                setattr(self, f"decode_conv_{decode_i}", decode_conv)
                setattr(self, f"decode_up_{decode_i}", decode_up)

            self.seg_output_conv = HighwayOutput3D(in_channels=decode_out_channels,
                                                   out_channels=seg_channels,
                                                   activation=seg_act)
        if self.get_class:
            if use_class_head_simple:
                self.classfication_head = ClassificationHeadSimple(feature_channel_num,
                                                                   class_channel,
                                                                   dropout_proba, class_act)
            else:
                self.classfication_head = ClassificationHead(feature_channel_num,
                                                             class_channel,
                                                             dropout_proba, class_act)

    def forward(self, input_tensor):
        output = []
        encode_feature = self.base_model(input_tensor)
        decoded = encode_feature
        if self.get_seg:
            for decode_i in range(0, 5):
                if self.skip_connect:
                    skip_connect_tensor = getattr(self.base_model,
                                                  f"skip_connect_tensor_{4 - decode_i}")
                    decoded = torch.cat([decoded,
                                         skip_connect_tensor], dim=1)
                decode_conv = getattr(self, f"decode_conv_{decode_i}")
                decode_up = getattr(self, f"decode_up_{decode_i}")
                decoded = decode_conv(decoded)
                decoded = decode_up(decoded)
            seg_output = self.seg_output_conv(decoded)
            output.append(seg_output)

        if self.get_class:
            class_output = self.classfication_head(encode_feature)
            output.append(class_output)
        if len(output) == 1:
            output = output[0]
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=USE_INPLACE)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ClassificationHeadSimple(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_proba, activation):
        super(ClassificationHeadSimple, self).__init__()
        self.gap_layer = nn.AdaptiveAvgPool3d((2, 2, 2))
        self.fc_1 = nn.Linear(in_channels * 8, in_channels)
        self.dropout_layer = nn.Dropout(p=dropout_proba, inplace=USE_INPLACE)
        self.relu_layer = nn.ReLU6(inplace=USE_INPLACE)
        self.fc_2 = nn.Linear(in_channels, num_classes)
        self.act = get_act(activation)

    def forward(self, x):
        x = self.gap_layer(x)
        x = x.flatten(start_dim=1, end_dim=-1)
        x = self.fc_1(x)
        x = self.dropout_layer(x)
        x = self.relu_layer(x)
        x = self.fc_2(x)
        x = self.act(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_proba, activation):
        super(ClassificationHead, self).__init__()
        self.in_channels = in_channels
        self.pos_encoder = PositionalEncoding(in_channels // 8)
        self.pixel_shffle = torch.nn.PixelShuffle(2)
        encoder_layers = TransformerEncoderLayer(d_model=in_channels // 8, nhead=8,
                                                 dropout=dropout_proba)
        self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                      num_layers=6)
        self.dropout = nn.Dropout(p=dropout_proba, inplace=USE_INPLACE)
        self.fc = nn.Linear(in_channels, num_classes)
        self.act = get_act(activation)

    def forward(self, x):
        batch_size, in_channel, z, h, w = x.shape
        # Assumes x has shape [N, C, H, W]
        x = self.pixel_shffle(x)
        # shape: [N, C // 8, Z * 2, H * 2,  W * 2]
        x = x.view(batch_size, self.in_channels // 8, -1)
        # shape: [N, C // 8, Z * H * W * 8]
        x = x.permute(0, 2, 1)
        # shape: [N, Z * H * W * 8, C // 8]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.dropout(x)
        # shape: [N, Z * H * W * 8, C // 8]
        x = x.view(batch_size, z * 2, h * 2, w * 2,
                   in_channel // 8)
        # shape: [N, Z * 2, H * 2, W * 2, C // 8]
        x = x.permute(0, 4, 1, 2, 3)
        # shape: [N, C // 8, Z * 2, H * 2, W * 2]
        x = space_to_depth_3d(x, 2)
        # shape: [N, C, Z, H, W]
        x = x.mean([2, 3, 4])
        # shape: [N, C]
        x = self.fc(x)
        x = self.act(x)
        return x

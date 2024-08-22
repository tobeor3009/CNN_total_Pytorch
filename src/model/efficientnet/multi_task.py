from segmentation_models_pytorch.encoders.efficientnet import EfficientNetEncoder

import torch
import math
from torch import nn
import numpy as np
from timm.models.layers import trunc_normal_
from ..inception_resnet_v2.common_module.layers import get_act, get_norm
from ..inception_resnet_v2.common_module.base_model import InceptionResNetV2_2D, get_skip_connect_channel_list
from ..inception_resnet_v2.common_module.layers import ConvBlock2D, AttentionPool
from ..inception_resnet_v2.common_module.layers_highway import MultiDecoder2D, HighwayOutput2D
from ..inception_resnet_v2.multi_task.common_layer import ClassificationHeadSimple
from ..inception_resnet_v2.multi_task.common_layer import ClassificationHead

DEFAULT_ACT = "silu"

class InceptionResNetV2MultiTask2D(nn.Module):
    def __init__(self, input_shape, class_channel=None, seg_channels=None, validity_shape=(1, 8, 8), inject_class_channel=None,
                 block_size=16, include_cbam=False, include_context=False, decode_init_channel=None,
                 skip_connect=True, dropout_proba=0.05, norm="batch", act=DEFAULT_ACT,
                 class_act="softmax", seg_act="sigmoid", validity_act="sigmoid",
                 get_seg=True, get_class=True, get_validity=False,
                 use_class_head_simple=True, use_seg_pixelshuffle_only=False
                 ):
        super().__init__()

        self.get_seg = get_seg
        self.get_class = get_class
        self.get_validity = get_validity
        self.inject_class_channel = inject_class_channel

        decode_init_channel = block_size * \
            64 if decode_init_channel is None else decode_init_channel
        input_shape = np.array(input_shape)
        n_input_channels, init_h, init_w = input_shape
        feature_hw = (init_h // (2 ** 5),
                      init_w // (2 ** 5))
        feature_h, feature_w = feature_hw

        feature_channel_num = block_size * 96
        self.feature_shape = np.array([feature_channel_num,
                                       input_shape[1] // 32,
                                       input_shape[2] // 32])
        self.skip_connect = skip_connect

        skip_connect_channel_list = get_skip_connect_channel_list(block_size)

        self.base_model = InceptionResNetV2_2D(n_input_channels=n_input_channels, block_size=block_size,
                                               padding="same", norm=norm, act=act,
                                               include_cbam=include_cbam, include_context=include_context,
                                               include_skip_connection_tensor=skip_connect)
        if self.get_seg:
            self.decode_init_conv = ConvBlock2D(in_channels=feature_channel_num,
                                                out_channels=decode_init_channel,
                                                kernel_size=1, norm=norm, act=act)
            for decode_i in range(0, 5):
                h, w = (init_h // (2 ** (5 - decode_i)),
                        init_w // (2 ** (5 - decode_i)))
                decode_in_channels = int(decode_init_channel //
                                         (2 ** decode_i))
                decode_out_channels = int(decode_init_channel //
                                          (2 ** (decode_i + 1)))
                if skip_connect:
                    skip_channel = skip_connect_channel_list[4 - decode_i]
                    decode_skip_conv = ConvBlock2D(in_channels=skip_channel,
                                                   out_channels=decode_in_channels,
                                                   kernel_size=1, norm=norm, act=act)
                    decode_in_channels *= 2
                    setattr(self,
                            f"decode_skip_conv_{decode_i}", decode_skip_conv)
                decode_conv = ConvBlock2D(in_channels=decode_in_channels,
                                          out_channels=decode_out_channels,
                                          kernel_size=3, norm=norm, act=act)
                decode_up = MultiDecoder2D(input_hw=(h, w),
                                           in_channels=decode_out_channels,
                                           out_channels=decode_out_channels,
                                           kernel_size=2, norm=norm, act=act,
                                           use_highway=False,
                                           use_pixelshuffle_only=use_seg_pixelshuffle_only)
                setattr(self, f"decode_conv_{decode_i}", decode_conv)
                setattr(self, f"decode_up_{decode_i}", decode_up)

            self.seg_output_conv = HighwayOutput2D(in_channels=decode_out_channels,
                                                   out_channels=seg_channels,
                                                   act=seg_act, use_highway=False)
        if self.get_class:
            if use_class_head_simple:
                self.classfication_head = ClassificationHeadSimple(feature_channel_num,
                                                                   class_channel,
                                                                   dropout_proba, class_act,
                                                                   mode="2d")
            else:
                self.classfication_head = ClassificationHead((feature_h, feature_w),
                                                             feature_channel_num,
                                                             class_channel,
                                                             dropout_proba, class_act)
        if get_validity:
            validity_init_channel = block_size * 32
            self.validity_conv_1 = ConvBlock2D(feature_channel_num, validity_init_channel,
                                               kernel_size=3, padding=1,
                                               norm="spectral", act=act)
            self.validity_conv_2 = ConvBlock2D(validity_init_channel,
                                               validity_init_channel // 2,
                                               kernel_size=3, padding=1,
                                               norm="spectral", act=act)
            self.validity_conv_3 = ConvBlock2D(validity_init_channel // 2,
                                               validity_init_channel // 2,
                                               kernel_size=3, padding=1,
                                               norm="spectral", act=act)
            self.validity_avg_pool = nn.AdaptiveAvgPool2d(validity_shape[1:])
            self.validity_final_conv = ConvBlock2D(validity_init_channel // 2, validity_shape[0],
                                                   kernel_size=1, act=validity_act, norm=None)
        if inject_class_channel is not None and get_seg:
            self.inject_linear = nn.Linear(inject_class_channel,
                                           decode_init_channel, bias=False)
            self.inject_norm = get_norm("layer", decode_init_channel, "2d")
            inject_pos_embed_shape = torch.zeros(1, 1,
                                                 *self.feature_shape[1:])
            self.inject_absolute_pos_embed = nn.Parameter(
                inject_pos_embed_shape)
            trunc_normal_(self.inject_absolute_pos_embed, std=.02)
            self.inject_cat_conv = nn.Conv2d(decode_init_channel * 2,
                                             decode_init_channel, kernel_size=1, padding=0, bias=False)

    def validity_forward(self, x):
        x = self.validity_conv_1(x)
        x = self.validity_conv_2(x)
        x = self.validity_conv_3(x)
        x = self.validity_avg_pool(x)
        x = self.validity_final_conv(x)
        return x

    def forward(self, input_tensor, inject_class=None):
        output = []
        encode_feature = self.base_model(input_tensor)
        if self.get_seg:
            decoded = encode_feature
            decoded = self.decode_init_conv(decoded)
            if self.inject_class_channel is not None:
                inject_class = self.inject_linear(inject_class)
                inject_class = self.inject_norm(inject_class)
                inject_class = inject_class[:, :, None, None]
                inject_class = inject_class.repeat(1, 1,
                                                   decoded.shape[2],
                                                   decoded.shape[3])
                inject_class = inject_class + self.inject_absolute_pos_embed
                decoded = torch.cat([decoded, inject_class], dim=1)
                decoded = self.inject_cat_conv(decoded)
            for decode_i in range(0, 5):
                if self.skip_connect:
                    skip_connect_tensor = getattr(self.base_model,
                                                  f"skip_connect_tensor_{4 - decode_i}")
                    skip_conv = getattr(self,
                                        f"decode_skip_conv_{decode_i}")
                    skip_connect_tensor = skip_conv(skip_connect_tensor)
                    decoded = torch.cat([decoded,
                                         skip_connect_tensor], axis=1)
                decode_conv = getattr(self, f"decode_conv_{decode_i}")
                decode_up = getattr(self, f"decode_up_{decode_i}")
                decoded = decode_conv(decoded)
                decoded = decode_up(decoded)
            seg_output = self.seg_output_conv(decoded)
            output.append(seg_output)

        if self.get_class:
            class_output = self.classfication_head(encode_feature)
            output.append(class_output)

        if self.get_validity:
            validity_output = self.validity_forward(encode_feature)
            output.append(validity_output)

        if len(output) == 1:
            output = output[0]
        if len(output) == 0:
            output = encode_feature
        return output

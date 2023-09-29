import torch
import math
from torch import nn
import numpy as np
from .layers import get_act
from .base_model import MiniInceptionResNetV2_2D, get_skip_connect_channel_list
from .layers import space_to_depth
from .layers import ConvBlock2D, Decoder2D, Output2D
from .multi_task import ClassificationHeadSimple, ClassificationHead
USE_INPLACE = True


class InceptionResNetV2Generation2D(nn.Module):
    def __init__(self, input_shape, class_channel, seg_channels, block_size=16,
                 include_cbam=False, include_context=False, decode_init_channel=768,
                 skip_connect=True, dropout_proba=0.05, class_act="softmax", seg_act="sigmoid",
                 get_seg=True, get_class=True, use_class_head_simple=True, use_seg_pixelshuffle=True
                 ):
        super().__init__()

        self.get_seg = get_seg
        self.get_class = get_class

        input_shape = np.array(input_shape)
        n_input_channels, init_h, init_w = input_shape
        feature_h, feature_w = (init_h // (2 ** 5),
                                init_w // (2 ** 5),)

        feature_channel_num = block_size * 32
        self.feature_shape = np.array([feature_channel_num,
                                       input_shape[1] // 32,
                                       input_shape[2] // 32])
        self.skip_connect = skip_connect

        skip_connect_channel_list = get_skip_connect_channel_list(block_size)

        self.base_model = MiniInceptionResNetV2_2D(n_input_channels=n_input_channels, block_size=block_size,
                                                   padding="same", include_cbam=include_cbam, include_context=include_context,
                                                   include_skip_connection_tensor=skip_connect)
        if self.get_seg:
            for decode_i in range(0, 3):
                h, w = (init_h // (2 ** (3 - decode_i)),
                        init_w // (2 ** (3 - decode_i)))
                decode_in_channels = decode_init_channel // (
                    2 ** (decode_i - 1)) if decode_i > 0 else feature_channel_num
                if skip_connect:
                    decode_in_channels += skip_connect_channel_list[2 - decode_i]
                decode_out_channels = decode_init_channel // (2 ** decode_i)
                decode_conv = ConvBlock2D(in_channels=decode_in_channels,
                                          out_channels=decode_out_channels, kernel_size=3)
                decode_up = Decoder2D(input_hw=(h, w),
                                      in_channels=decode_out_channels,
                                      out_channels=decode_out_channels, kernel_size=2,
                                      use_pixelshuffle=use_seg_pixelshuffle)
                setattr(self, f"decode_conv_{decode_i}", decode_conv)
                setattr(self, f"decode_up_{decode_i}", decode_up)

            self.seg_output_conv = Output2D(in_channels=decode_out_channels,
                                            out_channels=seg_channels,
                                            activation=seg_act)
        if self.get_class:
            if use_class_head_simple:
                self.classfication_head = ClassificationHeadSimple(feature_channel_num,
                                                                   class_channel,
                                                                   dropout_proba, class_act)
            else:
                self.classfication_head = ClassificationHead((feature_h, feature_w),
                                                             feature_channel_num,
                                                             class_channel,
                                                             dropout_proba, class_act)

    def forward(self, input_tensor):
        output = []
        encode_feature = self.base_model(input_tensor)
        decoded = encode_feature
        if self.get_seg:
            for decode_i in range(0, 3):

                if self.skip_connect:
                    skip_connect_tensor = getattr(self.base_model,
                                                  f"skip_connect_tensor_{2 -  decode_i}")
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
        if len(output) == 1:
            output = output[0]
        return output

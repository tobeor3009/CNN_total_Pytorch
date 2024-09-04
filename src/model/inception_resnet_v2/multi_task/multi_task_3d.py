import torch
import math
from torch import nn
import numpy as np
from timm.models.layers import trunc_normal_
from ..common_module.base_model import InceptionResNetV2_3D, get_skip_connect_channel_list
from ..common_module.layers import get_act, get_norm
from ..common_module.layers import space_to_depth_3d, DEFAULT_ACT
from ..common_module.layers import ConvBlock3D, Output3D
from ..common_module.layers_highway import MultiDecoder3D, HighwayOutput3D
from .common_layer import ClassificationHeadSimple
from .common_layer import ClassificationHead


class InceptionResNetV2MultiTask3D(nn.Module):
    def __init__(self, input_shape, class_channel=None, seg_channels=None, validity_shape=(1, 8, 8, 8),
                 inject_class_channel=None, block_size=16,
                 z_channel_preserve=False, decode_init_channel=None,
                 skip_connect=True, norm="batch", act=DEFAULT_ACT, dropout_proba=0.05,
                 seg_act="sigmoid", class_act="softmax", recon_act="sigmoid", validity_act="sigmoid",
                 get_seg=True, get_class=True, get_recon=False, get_validity=False,
                 use_class_head_simple=True,
                 use_decode_pixelshuffle_only=False, use_decode_simpleoutput=True
                 ):
        super().__init__()

        self.block_size = block_size
        self.feature_channel_num = block_size * 96
        self.get_seg = get_seg
        self.get_class = get_class
        self.get_recon = get_recon
        self.get_validity = get_validity
        self.inject_class_channel = inject_class_channel

        self.act = act
        self.dropout_proba = dropout_proba
        self.conv_block_common_arg_dict = {
            "norm": norm,
            "act": act,
            "dropout_proba": dropout_proba
        }
        self.channel_mode = "3d"
        if self.channel_mode == "2d":
            self.inject_unsqueeze = lambda x: x[:, :, None, None]
        elif self.channel_mode == "3d":
            self.inject_unsqueeze = lambda x: x[:, :, None, None, None]

        decode_init_channel = block_size * \
            64 if decode_init_channel is None else decode_init_channel
        input_shape = np.array(input_shape)
        input_channel = input_shape[0]
        feature_channel_num = block_size * 96
        self.feature_shape = np.array([feature_channel_num,
                                       *[shape_item // 32 for shape_item in input_shape[1:]]])
        self.skip_connect = skip_connect
        skip_connect_channel_list = get_skip_connect_channel_list(block_size)

        self.base_model = InceptionResNetV2_3D(n_input_channels=input_channel, block_size=block_size,
                                               padding="same", norm=norm, act=act, dropout_proba=dropout_proba,
                                               z_channel_preserve=z_channel_preserve, include_skip_connection_tensor=skip_connect)
        if self.get_seg:
            self.seg_module_list = self.get_decode_layers(input_shape, decode_init_channel,
                                   skip_connect_channel_list, z_channel_preserve,
                                   use_decode_pixelshuffle_only, use_decode_simpleoutput,
                                   seg_channels, seg_act)
        if self.get_class:
            if use_class_head_simple:
                self.classfication_head = ClassificationHeadSimple(feature_channel_num,
                                                                   class_channel,
                                                                   dropout_proba, class_act,
                                                                   mode="3d")
            else:
                self.classfication_head = ClassificationHead(self.feature_shape[1:],
                                                             feature_channel_num,
                                                             class_channel,
                                                             dropout_proba, class_act)
        if self.get_recon:
            self.recon_module_list = self.get_decode_layers(input_shape, decode_init_channel,
                                                            skip_connect_channel_list, z_channel_preserve, 
                                                            True, use_decode_simpleoutput, 
                                                            input_channel, recon_act)
        
        if get_validity:
            self.validity_block = self.get_validity_block(validity_shape, validity_act)

        if inject_class_channel is not None and get_seg:
            (self.inject_linear, self.inject_norm,
             self.inject_absolute_pos_embed, self.inject_cat_conv) = self.get_inject_layers(inject_class_channel,
                                                                                            decode_init_channel)
    def get_decode_layers(self, input_shape, decode_init_channel,
                          skip_connect_channel_list, z_channel_preserve,
                          use_decode_pixelshuffle_only, use_decode_simpleoutput,
                          seg_channels, seg_act):
        conv_block_common_arg_dict = self.conv_block_common_arg_dict
        init_conv = ConvBlock3D(in_channels=self.feature_channel_num,
                                out_channels=decode_init_channel, kernel_size=1, 
                                **conv_block_common_arg_dict)
        skip_conv_list = nn.ModuleList()
        conv_list = nn.ModuleList()
        up_list = nn.ModuleList()

        for decode_i in range(0, 5):
            decode_shape = input_shape[1:] // (2 ** (5 - decode_i))
            decode_in_channels = int(decode_init_channel //
                                        (2 ** decode_i))
            decode_out_channels = int(decode_in_channels // 2)
            if self.skip_connect:
                skip_channel = skip_connect_channel_list[4 - decode_i]
                decode_skip_conv = ConvBlock3D(in_channels=skip_channel,
                                                out_channels=decode_in_channels, kernel_size=1,
                                                **conv_block_common_arg_dict)
                decode_in_channels *= 2
                skip_conv_list.append(decode_skip_conv)
            decode_conv = ConvBlock3D(in_channels=decode_in_channels,
                                      out_channels=decode_out_channels, kernel_size=3,
                                      **conv_block_common_arg_dict)
            decode_kernel_size = (1, 2, 2) if z_channel_preserve else 2
            decode_up = MultiDecoder3D(input_zhw=decode_shape,
                                        in_channels=decode_out_channels,
                                        out_channels=decode_out_channels,
                                        kernel_size=decode_kernel_size,
                                        use_highway=False, use_pixelshuffle_only=use_decode_pixelshuffle_only,
                                        **conv_block_common_arg_dict)
            conv_list.append(decode_conv)
            up_list.append(decode_up)
            if use_decode_simpleoutput:
                output_conv = Output3D(in_channels=decode_out_channels,
                                                out_channels=seg_channels,
                                                act=seg_act)
            else:
                output_conv = HighwayOutput3D(in_channels=decode_out_channels,
                                                       out_channels=seg_channels,
                                                       act=seg_act, use_highway=False)
        return nn.ModuleList([init_conv, skip_conv_list, conv_list, up_list, output_conv])
    
    def get_inject_layers(self, inject_class_channel, decode_init_channel):
        inject_linear = nn.Linear(inject_class_channel,
                                        decode_init_channel, bias=False)
        inject_norm = get_norm("layer", decode_init_channel, self.channel_mode)
        inject_pos_embed_shape = torch.zeros(1, 1, *self.feature_shape[1:])
        inject_absolute_pos_embed = nn.Parameter(inject_pos_embed_shape)
        trunc_normal_(inject_absolute_pos_embed, std=.02)
        inject_cat_conv = ConvBlock3D(decode_init_channel * 2, decode_init_channel,
                                      kernel_size=1, padding=0, bias=False, act=None)
        return inject_linear, inject_norm, inject_absolute_pos_embed, inject_cat_conv
    
    def get_validity_block(self, validity_shape, validity_act):
        validity_init_channel = self.block_size * 32
        common_arg_dict = {
            "norm": "spectral", 
            "act": self.act, 
            "dropout_proba": self.dropout_proba
        }

        validity_conv_1 = ConvBlock3D(self.feature_channel_num, validity_init_channel,
                                      kernel_size=3, padding=1,
                                      **common_arg_dict)
        validity_conv_2 = ConvBlock3D(validity_init_channel,
                                      validity_init_channel // 2,
                                      kernel_size=3, padding=1,
                                      **common_arg_dict)
        validity_conv_3 = ConvBlock3D(validity_init_channel // 2,
                                      validity_init_channel // 2,
                                      kernel_size=3, padding=1,
                                      **common_arg_dict)
        validity_avg_pool = nn.AdaptiveAvgPool3d(validity_shape[1:])
        validity_final_conv = ConvBlock3D(validity_init_channel // 2, validity_shape[0],
                                          kernel_size=1, act=validity_act, norm=None)
        validity_block = nn.Sequential(
            validity_conv_1,
            validity_conv_2,
            validity_conv_3,
            validity_avg_pool,
            validity_final_conv,
        )
        return validity_block
    
    def forward_inject_class_tensor(self, decoded, inject_class):
        if inject_class is not None:
            inject_class = self.inject_linear(inject_class)
            inject_class = self.inject_norm(inject_class)
            inject_class = self.inject_unsqueeze(inject_class)
            inject_class = inject_class.repeat(1, 1, *decoded.shape[2:])
            inject_class = inject_class + self.inject_absolute_pos_embed
            decoded = torch.cat([decoded, inject_class], dim=1)
            decoded = self.inject_cat_conv(decoded)
        return decoded
    
    def forward_decode_block(self, encode_feature, inject_class,
                             init_conv, skip_conv_list, conv_list, 
                             up_list, output_conv):
        decoded = init_conv(encode_feature)
        decoded = self.forward_inject_class_tensor(decoded, inject_class)
        
        for decode_i in range(0, 5):
            if self.skip_connect:
                skip_connect_tensor = getattr(self.base_model, f"skip_connect_tensor_{4 - decode_i}")
                skip_conv = skip_conv_list[decode_i]
                skip_connect_tensor = skip_conv(skip_connect_tensor)
                decoded = torch.cat([decoded, skip_connect_tensor], axis=1)
            decode_conv = conv_list[decode_i]
            decode_up = up_list[decode_i]
            decoded = decode_conv(decoded)
            decoded = decode_up(decoded)
        seg_output = output_conv(decoded)
        return seg_output

    def forward(self, input_tensor, inject_class=None):
        output = []
        encode_feature = self.base_model(input_tensor)
        if self.get_seg:
            seg_output = self.forward_decode_block(encode_feature, inject_class,
                                                   *self.seg_module_list)
            output.append(seg_output)

        if self.get_class:
            class_output = self.classfication_head(encode_feature)
            output.append(class_output)

        if self.get_recon:
            recon_output = self.forward_decode_block(encode_feature, inject_class,
                                                   *self.recon_module_list)
            output.append(recon_output)

        if self.get_validity:
            validity_output = self.validity_block(encode_feature)
            output.append(validity_output)

        if len(output) == 1:
            output = output[0]
        if len(output) == 0:
            output = encode_feature
        return output

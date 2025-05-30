import torch
from torch import nn
import numpy as np
from functools import partial
from timm.models.layers import trunc_normal_
from ..common_module.layers import get_act, get_norm
from ..common_module.base_model_v2 import InceptionResNetV2_2D, get_skip_connect_channel_list
from ..common_module.layers import DEFAULT_ACT
from ..common_module.layers import ConvBlock2D, Output2D_V2
from ..common_module.layers_highway import MultiDecoder2D_V2, ConvTransposeDecoder2D_V2, HighwayOutput2D
from .common_layer import ClassificationHeadSimple
from .common_layer import ClassificationHead
from src.model.train_util.common import process_with_checkpoint
class InceptionResNetV2MultiTask2D(nn.Module):
    def __init__(self, input_shape, class_channel=None, seg_channels=None, validity_shape=(1, 8, 8), inject_class_channel=None,
                 block_size=16, include_cbam=False, decode_init_channel=None,
                 norm="instance", act=DEFAULT_ACT, dropout_proba=0.05,
                 seg_act="softmax", class_act="softmax", recon_act="sigmoid", validity_act="sigmoid",
                 get_seg=True, get_class=True, get_recon=False, get_validity=False,
                 use_class_head_simple=True, include_upsample=False,
                 use_decode_simpleoutput=True, use_seg_conv_transpose=True,
                 use_checkpoint=False
                 ):
        super().__init__()

        self.block_size = block_size
        self.feature_channel_num = block_size * 96
        self.get_seg = get_seg
        self.get_class = get_class
        self.get_recon = get_recon
        self.get_validity = get_validity
        self.inject_class_channel = inject_class_channel
        self.process_with_checkpoint = partial(process_with_checkpoint, use_checkpoint=use_checkpoint)


        self.act = act
        self.dropout_proba = dropout_proba
        self.conv_block_common_arg_dict = {
            "norm": norm,
            "act": act,
            "dropout_proba": dropout_proba
        }


        self.channel_mode = "2d"
        if self.channel_mode == "2d":
            self.inject_unsqueeze = lambda x: x[:, :, None, None]
        elif self.channel_mode == "3d":
            self.inject_unsqueeze = lambda x: x[:, :, None, None, None]

        decode_init_channel = block_size * 64 \
            if decode_init_channel is None else decode_init_channel
        input_shape = np.array(input_shape)
        input_channel = input_shape[0]
        feature_channel_num = block_size * 96
        self.feature_shape = np.array([feature_channel_num,
                                       *[shape_item // 32 for shape_item in input_shape[1:]]])
        skip_connect_channel_list = get_skip_connect_channel_list(block_size)

        self.base_model = InceptionResNetV2_2D(n_input_channels=input_channel, block_size=block_size,
                                               padding="same", norm=norm, act=act, dropout_proba=dropout_proba,
                                               include_cbam=include_cbam, include_skip_connection_tensor=True,
                                               use_checkpoint=use_checkpoint)
        if self.get_seg:
            if use_seg_conv_transpose:
                seg_decode_mode = "conv_transpose"
            else:
                seg_decode_mode = "upsample_pixelshuffle"
            self.seg_module_list = self.get_decode_layers(decode_init_channel,
                                                        skip_connect_channel_list, include_upsample,
                                                        use_decode_simpleoutput, seg_channels, seg_act, seg_decode_mode)
        if self.get_class:
            if use_class_head_simple:
                self.classfication_head = ClassificationHeadSimple(feature_channel_num,
                                                                   class_channel,
                                                                   dropout_proba, class_act,
                                                                   mode="2d")
            else:
                self.classfication_head = ClassificationHead(self.feature_shape[1:],
                                                             feature_channel_num,
                                                             class_channel,
                                                             dropout_proba, class_act)
        if self.get_recon:
            recon_decode_mode = "upsample_pixelshuffle"
            self.recon_module_list = self.get_decode_layers(decode_init_channel,
                                                            skip_connect_channel_list, include_upsample,
                                                            use_decode_simpleoutput, input_channel,
                                                            recon_act, recon_decode_mode)
        if get_validity:
            self.validity_block = self.get_validity_block(validity_shape, validity_act)
        if inject_class_channel is not None and get_seg:
            (self.inject_linear, self.inject_norm,
             self.inject_absolute_pos_embed, self.inject_cat_conv) = self.get_inject_layers(inject_class_channel,
                                                                                            decode_init_channel)
        
    def get_decode_layers(self, decode_init_channel,
                          skip_connect_channel_list, include_upsample,
                          use_decode_simpleoutput, decode_channels, decode_act, decode_mode):
        conv_block_common_arg_dict = self.conv_block_common_arg_dict
        init_conv = ConvBlock2D(in_channels=self.feature_channel_num,
                                out_channels=decode_init_channel, kernel_size=1,
                                **conv_block_common_arg_dict)
        up_list = nn.ModuleList()
        conv_list = nn.ModuleList()
        
        for decode_i in range(0, 5):
            skip_channels = skip_connect_channel_list[4 - decode_i]
            decode_in_channels = int(decode_init_channel // (2 ** decode_i))
            decode_out_channels = decode_in_channels // 2
            if decode_mode == "conv_transpose":
                decode_up = ConvTransposeDecoder2D_V2(in_channels=decode_in_channels,
                                                      skip_channels=skip_channels,
                                                      out_channels=decode_out_channels, kernel_size=2,
                                                      **conv_block_common_arg_dict)
            elif decode_mode == "upsample_pixelshuffle":
                decode_up = MultiDecoder2D_V2(in_channels=decode_in_channels,
                                              skip_channels=skip_channels,
                                            out_channels=decode_out_channels,
                                            kernel_size=2, use_highway=False,
                                            include_upsample=include_upsample,
                                            include_conv_transpose=not include_upsample,
                                            **conv_block_common_arg_dict)

            decode_conv = ConvBlock2D(in_channels=decode_out_channels,
                                        out_channels=decode_out_channels,
                                        kernel_size=3, **conv_block_common_arg_dict)
            up_list.append(decode_up)
            conv_list.append(decode_conv)

        if use_decode_simpleoutput:
            output_conv = Output2D_V2(in_channels=decode_out_channels,
                                    out_channels=decode_channels,
                                    act=decode_act)
        else:
            output_conv = HighwayOutput2D(in_channels=decode_out_channels,
                                            out_channels=decode_channels,
                                            act=decode_act, use_highway=False)
        return nn.ModuleList([init_conv, up_list, conv_list, output_conv])

    def get_inject_layers(self, inject_class_channel, decode_init_channel):
        inject_linear = nn.Linear(inject_class_channel,
                                  decode_init_channel, bias=False)
        inject_norm = get_norm("layer", decode_init_channel, self.channel_mode)
        inject_pos_embed_shape = torch.zeros(1, 1, *self.feature_shape[1:])
        inject_absolute_pos_embed = nn.Parameter(inject_pos_embed_shape)
        trunc_normal_(inject_absolute_pos_embed, std=.02)
        inject_cat_conv = ConvBlock2D(decode_init_channel * 2,
                                      decode_init_channel, kernel_size=1, padding=0, bias=False, act=None)
        return inject_linear, inject_norm, inject_absolute_pos_embed, inject_cat_conv

    def get_validity_block(self, validity_shape, validity_act):
        validity_init_channel = self.block_size * 32
        common_arg_dict = {
            "norm": "spectral", 
            "act": self.act, 
            "dropout_proba": self.dropout_proba
        }
        validity_conv_1 = ConvBlock2D(self.feature_channel_num, validity_init_channel,
                                        kernel_size=3, padding=1,
                                        **common_arg_dict)
        validity_conv_2 = ConvBlock2D(validity_init_channel,
                                            validity_init_channel // 2,
                                            kernel_size=3, padding=1,
                                            **common_arg_dict)
        validity_conv_3 = ConvBlock2D(validity_init_channel // 2,
                                            validity_init_channel // 2,
                                            kernel_size=3, padding=1,
                                            **common_arg_dict)
        validity_avg_pool = nn.AdaptiveAvgPool2d(validity_shape[1:])
        validity_final_conv = ConvBlock2D(validity_init_channel // 2, validity_shape[0],
                                                kernel_size=1, act=validity_act, norm=None, dropout_proba=0.0)
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
    
    def forward_decode_block(self, encode_feature, inject_class, skip_list,
                             init_conv, up_list, conv_list, output_conv):
        decoded = init_conv(encode_feature)
        decoded = self.forward_inject_class_tensor(decoded, inject_class)
        
        for decode_i in range(0, 5):
            decode_up = up_list[decode_i]
            decode_conv = conv_list[decode_i]
            skip_connect_tensor = skip_list[decode_i]
            decoded = self.process_with_checkpoint(decode_up, decoded, skip_connect_tensor)
            decoded = self.process_with_checkpoint(decode_conv, decoded)
        seg_output = self.process_with_checkpoint(output_conv, decoded)
        return seg_output
    
    def forward(self, input_tensor, inject_class=None):
        output = {
            "pred": None,
            "seg_pred": None,
            "class_pred": None,
            "recon_pred": None,
            "validity_pred": None
        }
        encode_feature_list = self.base_model(input_tensor)
        encode_feature = encode_feature_list.pop()
        skip_list = encode_feature_list[::-1]
        if self.get_seg:
            seg_output = self.forward_decode_block(encode_feature, inject_class, skip_list,
                                                   *self.seg_module_list)
            output["seg_pred"] = seg_output

        if self.get_class:
            class_output = self.process_with_checkpoint(self.classfication_head, encode_feature)
            output["class_pred"] = class_output

        if self.get_recon:
            recon_output = self.forward_decode_block(encode_feature, inject_class, skip_list,
                                                   *self.recon_module_list)
            output["recon_pred"] = recon_output

        if self.get_validity:
            validity_output = self.process_with_checkpoint(self.validity_block, encode_feature)
            output["validity_pred"] = validity_output
        
        return output

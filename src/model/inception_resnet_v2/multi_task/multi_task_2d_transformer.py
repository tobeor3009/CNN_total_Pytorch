import torch
import math
from torch import nn
import numpy as np
from timm.models.layers import trunc_normal_
from ..common_module.layers import get_act, get_norm
from ..common_module.base_model import InceptionResNetV2_2D, get_skip_connect_channel_list
from ..common_module.transformer_layers import PositionalEncoding
from ..common_module.layers import space_to_depth, DEFAULT_ACT
from ..common_module.layers import ConvBlock2D, AttentionPool, ConvBlock1D
from ..common_module.layers_highway import MultiDecoder2D, HighwayOutput2D
from ...swin_transformer.model_2d.swin_layers import PatchEmbed, BasicLayerV2
from ...swin_transformer.model_2d.swin_layers import PatchExpanding, PatchExpandingConcat
USE_INPLACE = True


class InceptionResNetV2MultiTask2D(nn.Module):
    def __init__(self, input_shape, class_channel=None, seg_channels=None, validity_shape=(1, 8, 8), inject_class_channel=None,
                 block_size=16, include_cbam=False, include_context=False, decode_init_channel=None,
                 skip_connect=True, dropout_proba=0.05,
                 conv_norm="instance", conv_act=DEFAULT_ACT,
                 trans_norm=nn.LayerNorm, trans_act=DEFAULT_ACT,
                 patch_size=4, depths=[2, 2, 2, 2, 2], num_heads=[4, 2, 2, 1, 1],
                 window_sizes=[2, 2, 2, 4, 4], mlp_ratio=4.0,
                 class_act="softmax", seg_act="sigmoid", validity_act="sigmoid",
                 get_seg=True, get_class=True, get_validity=False,
                 decoder_simple=True, use_class_head_simple=True
                 ):
        super().__init__()
        if decoder_simple:
            expand_block = PatchExpanding
        else:
            expand_block = PatchExpandingConcat
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
        embed_num = np.prod(self.feature_shape[1:] // patch_size)
        self.skip_connect = skip_connect

        skip_connect_channel_list = get_skip_connect_channel_list(block_size)

        self.base_model = InceptionResNetV2_2D(n_input_channels=n_input_channels, block_size=block_size,
                                               padding="same", norm=conv_norm, act=conv_act,
                                               include_cbam=include_cbam, include_context=include_context,
                                               include_skip_connection_tensor=skip_connect)
        if self.get_seg:
            self.decode_init_embed = PatchEmbed(img_size=feature_hw, patch_size=patch_size,
                                                in_chans=feature_channel_num,
                                                embed_dim=decode_init_channel,
                                                norm_layer=trans_norm)
            for decode_i in range(0, 5):
                down_ratio = 2 ** (5 - decode_i)
                channel_down_ratio = 2 ** decode_i
                h, w = (init_h // down_ratio,
                        init_w // down_ratio)
                resolution_2d = (init_h // down_ratio // patch_size,
                                 init_w // down_ratio // patch_size)
                decode_in_channels = int(decode_init_channel //
                                         channel_down_ratio)
                if skip_connect:
                    skip_channel = skip_connect_channel_list[4 - decode_i]
                    decode_skip_embed = PatchEmbed(img_size=(h, w), patch_size=patch_size,
                                                   in_chans=skip_channel,
                                                   embed_dim=decode_in_channels,
                                                   norm_layer=trans_norm)
                    skip_conv = ConvBlock1D(in_channels=decode_in_channels * 2,
                                            out_channels=decode_in_channels,
                                            kernel_size=1, channel_last=True)
                    setattr(self,
                            f"decode_skip_embed_{decode_i}", decode_skip_embed)
                    setattr(self,
                            f"decode_skip_conv_{decode_i}", skip_conv)
                decode_up_trans = BasicLayerV2(dim=decode_in_channels,
                                               input_resolution=resolution_2d,
                                               depth=depths[decode_i],
                                               num_heads=num_heads[decode_i],
                                               window_size=window_sizes[decode_i],
                                               mlp_ratio=mlp_ratio,
                                               qkv_bias=True,
                                               drop=0.0, attn_drop=0.0,
                                               drop_path=0.,
                                               norm_layer=trans_norm,
                                               upsample=expand_block)
                setattr(self, f"decode_up_trans_{decode_i}", decode_up_trans)
            resolution_2d = np.array(resolution_2d) * 2
            decode_out_channels = decode_in_channels // 2
            self.seg_final_expanding = expand_block(input_resolution=resolution_2d,
                                                    dim=decode_out_channels,
                                                    return_vector=False,
                                                    dim_scale=patch_size,
                                                    norm_layer=trans_norm
                                                    )
            self.seg_final_conv = nn.Conv2d(decode_out_channels // 2, seg_channels,
                                            kernel_size=1, padding=0)
            self.seg_final_act = get_act(seg_act)
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
        if get_validity:
            validity_init_channel = block_size * 32
            patch_hw = np.array(feature_hw) // patch_size
            self.validity_embed = PatchEmbed(img_size=feature_hw, patch_size=patch_size,
                                             in_chans=feature_channel_num,
                                             embed_dim=validity_init_channel,
                                             norm_layer=trans_norm)
            self.validity_block_1 = BasicLayerV2(dim=validity_init_channel,
                                                 input_resolution=patch_hw,
                                                 depth=depths[0],
                                                 num_heads=num_heads[0],
                                                 window_size=window_sizes[0],
                                                 mlp_ratio=mlp_ratio,
                                                 qkv_bias=True,
                                                 drop=0.0, attn_drop=0.0,
                                                 drop_path=0.,
                                                 norm_layer=trans_norm)
            self.validity_block_2 = BasicLayerV2(dim=validity_init_channel,
                                                 input_resolution=patch_hw,
                                                 depth=depths[1],
                                                 num_heads=num_heads[1],
                                                 window_size=window_sizes[1],
                                                 mlp_ratio=mlp_ratio,
                                                 qkv_bias=True,
                                                 drop=0.0, attn_drop=0.0,
                                                 drop_path=0.,
                                                 norm_layer=trans_norm)
            self.validity_final_expanding = expand_block(input_resolution=patch_hw,
                                                         dim=validity_init_channel,
                                                         return_vector=False,
                                                         dim_scale=patch_size,
                                                         norm_layer=trans_norm
                                                         )
            self.validity_avg_pool = nn.AdaptiveAvgPool2d(validity_shape[1:])
            self.validity_final_conv = ConvBlock2D(validity_init_channel // 2, validity_shape[0],
                                                   kernel_size=1, act=validity_act, norm=None)
        if inject_class_channel is not None and get_seg:
            self.inject_linear = nn.Linear(inject_class_channel,
                                           decode_init_channel, bias=False)
            self.inject_norm = get_norm("layer", decode_init_channel, "2d")
            inject_pos_embed_shape = torch.zeros(1, embed_num,
                                                 decode_init_channel)
            self.inject_absolute_pos_embed = nn.Parameter(
                inject_pos_embed_shape)
            trunc_normal_(self.inject_absolute_pos_embed, std=.02)
            self.inject_cat_conv = ConvBlock1D(decode_init_channel * 2,
                                               decode_init_channel,
                                               kernel_size=1, padding=0, bias=False,
                                               norm=None, act=trans_act,
                                               channel_last=True)

    def validity_forward(self, x):
        x = self.validity_embed(x)
        x = self.validity_block_1(x)
        x = self.validity_block_2(x)
        x = self.validity_final_expanding(x)
        x = self.validity_avg_pool(x)
        x = self.validity_final_conv(x)
        return x

    def forward(self, input_tensor, inject_class=None):
        output = []
        encode_feature = self.base_model(input_tensor)
        if self.get_seg:
            decoded = encode_feature
            decoded = self.decode_init_embed(decoded)
            if self.inject_class_channel is not None:
                inject_class = self.inject_linear(inject_class)
                inject_class = self.inject_norm(inject_class)
                inject_class = inject_class[:, None, :]
                inject_class = inject_class.repeat(1, decoded.shape[1], 1)
                inject_class = inject_class + self.inject_absolute_pos_embed
                decoded = torch.cat([decoded, inject_class], dim=-1)
                decoded = self.inject_cat_conv(decoded)
            for decode_i in range(0, 5):
                if self.skip_connect:
                    skip_connect_tensor = getattr(self.base_model,
                                                  f"skip_connect_tensor_{4 - decode_i}")
                    decode_skip_embed = getattr(self,
                                                f"decode_skip_embed_{decode_i}")
                    skip_conv = getattr(self,
                                        f"decode_skip_conv_{decode_i}")
                    skip_connect_tensor = decode_skip_embed(
                        skip_connect_tensor)
                    decoded = torch.cat([decoded,
                                         skip_connect_tensor], axis=-1)
                    decoded = skip_conv(decoded)
                decode_up_trans = getattr(self, f"decode_up_trans_{decode_i}")
                decoded = decode_up_trans(decoded)
            seg_output = self.seg_final_expanding(decoded)
            seg_output = self.seg_final_conv(seg_output)
            seg_output = self.seg_final_act(seg_output)
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
        self.gap_layer = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_1 = nn.Linear(in_channels, in_channels // 2)
        self.dropout_layer = nn.Dropout(p=dropout_proba, inplace=USE_INPLACE)
        self.relu_layer = nn.ReLU6(inplace=USE_INPLACE)
        self.fc_2 = nn.Linear(in_channels // 2, num_classes)
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
    def __init__(self, feature_hw, in_channels, num_classes, dropout_proba, activation):
        super(ClassificationHead, self).__init__()
        self.attn_pool = AttentionPool(feature_num=np.prod(feature_hw), embed_dim=in_channels,
                                       num_heads=4, output_dim=in_channels * 2)
        self.dropout = nn.Dropout(p=dropout_proba, inplace=USE_INPLACE)
        self.fc = nn.Linear(in_channels * 2, num_classes)
        self.act = get_act(activation)

    def forward(self, x):
        x = self.attn_pool(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.act(x)
        return x

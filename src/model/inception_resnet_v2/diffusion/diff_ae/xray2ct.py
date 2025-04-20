from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.nn import init
from .nn import timestep_embedding
from .diffusion_layer import default, Return
from .diffusion_layer import ConvBlockND, ResNetBlockND, OutputND
from .diffusion_layer import ResNetBlockNDSkip, ConvBlockNDSkip, MultiDecoderND_V3
from .diffusion_layer import get_maxpool_nd, get_avgpool_nd
from .diffusion_layer import LinearAttention, Attention, AttentionBlock, MaxPool2d, AvgPool2d, MultiInputSequential
from .diffusion_layer import default, prob_mask_like, LearnedSinusoidalPosEmb, SinusoidalPosEmb, GroupNorm32
from .diffusion_layer import feature_z_normalize, z_normalize
from .diffusion_layer import ClassificationHeadSimple
from .sub_models import MLPSkipNet, Classifier
from src.model.inception_resnet_v2.common_module.layers import get_act, get_norm
from einops import rearrange, repeat
def get_skip_connect_channel_list(block_size, mini=False):
    if mini:
        return np.array([block_size * 2, block_size * 4, block_size * 12])
    else:
        return np.array([block_size * 8, block_size * 8, block_size * 8, block_size * 12,
                        block_size * 68, block_size * 130])

def get_encode_feature_channel(block_size, last_channel_ratio):
    feature_channel = block_size * 130 * last_channel_ratio
    return int(round(feature_channel))

def get_time_emb_dim(block_size):
    # emb_dim = block_size * 16
    # I found 512 is best size for all size
    emb_dim = 512
    time_emb_dim_init = emb_dim // 2
    time_emb_dim = emb_dim * 4
    return emb_dim, time_emb_dim_init, time_emb_dim

class InceptionResNetV2_UNet(nn.Module):
    def __init__(self, in_channel=3, img_size=256, block_size=8,
                 decode_init_channel=None, block_depth_info="middle",
                 norm=GroupNorm32, act="silu", num_class_embeds=None, drop_prob=0.05, cond_drop_prob=0.5,
                 self_condition=False, use_checkpoint=[False, False, False, False, True],
                 attn_info_list=[None, None, None, None, True], attn_dim_head=32, num_head_list=[1, 1, 1, 1, 1],
                 diffusion_out_channel=1, diffusion_act=None, diffusion_decode_fn_str_list=["conv_transpose", "pixel_shuffle"],
                use_residual_conv=True):
        super().__init__()
        last_channel_ratio = 1
        self.use_inception_block_attn = True
        if decode_init_channel is None:
            decode_init_channel = block_size * 96
        self.decode_init_channel = decode_init_channel
        if isinstance(block_depth_info, str):
            if block_depth_info == "tiny":
                block_depth_list = [1, 2, 1]
            elif block_depth_info == "mini":
                block_depth_list = [2, 4, 2]
            elif block_depth_info == "middle":
                block_depth_list = [5, 10, 5]
            elif block_depth_info == "large":
                block_depth_list = [10, 20, 10]
        elif isinstance(block_depth_info, int):
            block_depth_list = np.array([1, 2, 1]) * block_depth_info
        else:
            block_depth_list = block_depth_info
        self.block_depth_list = block_depth_list

        # for compability with diffusion_sample
        self.img_size = img_size
        self.out_channel = diffusion_out_channel
        self.self_condition = self_condition
        if self.self_condition:
            in_channel *= 2
        ##################################
        self.padding_3x3 = 1
        self.in_channel = in_channel
        self.block_size = block_size
        self.norm = norm
        self.act = act
        if isinstance(use_checkpoint, bool):
            use_checkpoint = [use_checkpoint for _ in attn_info_list]
        if isinstance(attn_dim_head, int):
            attn_dim_head = [attn_dim_head for _ in attn_info_list]
        self.use_checkpoint = use_checkpoint
        self.attn_info_list = attn_info_list
        self.attn_dim_head = attn_dim_head
        self.num_head_list = num_head_list
        self.drop_prob = drop_prob
        self.cond_drop_prob = cond_drop_prob
        self.block_scale_list = [0.17, 0.1, 0.2]
        self.act_layer = get_act(act)
        img_dim = 3
        self.img_dim = 3
        ##################################
        assert len(attn_info_list) == 5, "check len(attn_info_list) == 5"
        assert len(use_checkpoint) == 5, "check len(use_checkpoint) == 5"
        assert len(num_head_list) == 5, "check len(num_head_list) == 5"
        assert isinstance(img_dim, int), "img_dim must be int"
        ##################################
        if use_residual_conv:
            conv_block = ResNetBlockND
            conv_skip_block = ResNetBlockNDSkip
        else:
            conv_block = ConvBlockND
            conv_skip_block = ConvBlockNDSkip
        self.use_residual_conv = use_residual_conv
        self.conv_block = conv_block
        self.conv_skip_block = conv_skip_block
        self.image_shape = self.get_image_init_shape()
        self.skip_channel_list = get_skip_connect_channel_list(self.block_size)
        self.model_depth = len(self.skip_channel_list) - 1
        ##################################
        emb_dim, time_emb_dim_init, time_emb_dim = get_time_emb_dim(block_size)
        self.time_emb_dim_init = time_emb_dim_init
        emb_dim_list = []
        emb_type_list = []
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim_init, time_emb_dim),
            get_act(act),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        emb_dim_list.append(time_emb_dim)
        emb_type_list.append("seq")

        # 1. time embedding is default setting
        # 2. if include_encoder, then add it to emb_dim_list, emb_type_list
        # 2. if num_class_embeds is not None, then add it to emb_dim_list, emb_type_list
        self.num_class_embeds = num_class_embeds
        if num_class_embeds is not None:
            class_emb_dim = emb_dim * 4
            self.class_emb_layer = nn.Embedding(num_class_embeds, class_emb_dim)
            if self.cond_drop_prob > 0:
                self.null_class_emb = nn.Parameter(torch.randn(class_emb_dim))
            else:
                self.null_class_emb = None
            
            self.class_mlp = nn.Sequential(
                nn.Linear(class_emb_dim, class_emb_dim),
                get_act(act),
                nn.Linear(class_emb_dim, class_emb_dim)
            )
            emb_dim_list.append(time_emb_dim)
            emb_type_list.append("cond")
        else:
            self.class_emb_layer = None
            self.null_class_emb = None
            self.class_mlp = None
        
        self.emb_dim_list = emb_dim_list
        self.emb_type_list = emb_type_list
        self.cond_drop_prob = cond_drop_prob
        self.feature_channel = get_encode_feature_channel(block_size, last_channel_ratio)

        self.set_encoder()
        self.diffusion_decoder_list = self.get_decoder(diffusion_out_channel, diffusion_act,
                                                        decode_fn_str_list=diffusion_decode_fn_str_list, is_diffusion=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, t=None, t_cond=None,
                x_start=None, cond=None, latent_feature=None,
                x_self_cond=None, class_labels=None, cond_drop_prob=None,
                infer_diffusion=True):
        output = None
        class_emb = self.process_class_emb(x, class_labels, cond_drop_prob)
        
        output = self._forward_diffusion(x=x, t=t, t_cond=t_cond, x_start=x_start, cond=cond,
                                        latent_feature=latent_feature, x_self_cond=x_self_cond, class_emb=class_emb)
        return output
    
    def _forward_diffusion(self, x, t, t_cond=None,
                            x_start=None, cond=None, latent_feature=None,
                            x_self_cond=None, class_emb=None):
        output = Return()
        emb_list = []
        # autoencoder original source not used t_cond varaiable
        if t_cond is None:
            t_cond = t
        time_emb = timestep_embedding(t, self.time_emb_dim_init)
        time_emb = self.time_mlp(time_emb)
        emb_list.append(time_emb)
        if (cond is None) and (latent_feature is None) and self.include_encoder:
            assert x.shape == x_start.shape, f"x.shape: {x.shape}, x_start.shape: {x_start.shape}"
            latent_feature = self.encoder(x_start)
        else:
            latent_feature = latent_feature or cond
        emb_list.append(latent_feature)
        if class_emb is not None:
            emb_list.append(class_emb)
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        diff_encode_feature, diff_skip_connect_list = self.encode_forward(x, *emb_list)
        diff_decode_feature = self.decode_forward(self.diffusion_decoder_list, diff_encode_feature, diff_skip_connect_list, *emb_list)
        output["pred"] = diff_decode_feature
        return output
    
    def process_class_emb(self, x, class_labels, cond_drop_prob):
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        if self.num_class_embeds is not None:
            batch, device = x.size(0), x.device
            class_emb = self.class_emb_layer(class_labels).to(dtype=x.dtype)
            if cond_drop_prob > 0:
                keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device=device)
                null_classes_emb = repeat(self.null_class_emb, 'd -> b d', b=batch)

                class_emb = torch.where(
                    rearrange(keep_mask, 'b -> b 1'),
                    class_emb, null_classes_emb
                )
            class_emb = self.class_mlp(class_emb)
        else:
            class_emb = None
        return class_emb
    
    def encode_forward(self, x, *args):
        # skip connection name list
        # ["stem_layer_1", "stem_layer_4", "stem_layer_7", "mixed_6a", "mixed_7a"]
        skip_connect_list = []
        stem = x
        for idx, (_, layer) in enumerate(self.stem.items()):
            stem = layer(stem, *args)
            if idx in [1, 3, 6, 9]:
                skip_connect_list.append(stem)
        # mixed_5b
        mixed_5b = self.process_encode_block(self.mixed_5b, self.mixed_5b_attn, stem, *args)
        # block_35
        block_35 = self.process_inception_block(self.block_35_mixed, self.block_35_up,
                                                mixed_5b, self.block_scale_list[0], *args)
        # mixed_6a: skip connect target
        mixed_6a = self.process_encode_block(self.mixed_6a, self.mixed_6a_attn, block_35, *args)
        skip_connect_list.append(mixed_6a)
        # block_17
        block_17 = self.process_inception_block(self.block_17_mixed, self.block_17_up,
                                                mixed_6a, self.block_scale_list[1], *args)
        # mixed_7a: skip connect target
        mixed_7a = self.process_encode_block(self.mixed_7a, self.mixed_7a_attn, block_17, *args)
        skip_connect_list.append(mixed_7a)
        # block_8
        block_8 = self.process_inception_block(self.block_8_mixed, self.block_8_up,
                                               mixed_7a, self.block_scale_list[2], *args)
        # final_output
        output = self.encode_final_block(block_8, *args)

        return output, skip_connect_list[::-1]
    
    def process_encode_block(self, block, attn, x, *args):
        output = []
        for (_, layer_list) in block.items():
            output_part = x
            for layer in layer_list:
                output_part = layer(output_part, *args)
            output.append(output_part)
        output = torch.cat(output, dim=1)
        if attn is not None:
            output = attn(output)
        return output

    def process_inception_block(self, block_mixed, block_up,
                                x, scale, *args):
        for mixed, up_layer in zip(block_mixed, block_up):
            x_temp = self.process_encode_block(mixed, None, x, *args)
            x_temp = up_layer(x_temp, *args)
            x = x + x_temp * scale
            x = self.act_layer(x)
        return x
        
    def decode_forward(self, decoder_list, encode_feature, skip_connect_list, *args):
        decode_layer_up_list, decode_block_list, decode_final_conv = decoder_list
        decode_feature = encode_feature
        for decode_idx, (decode_layer_up, decode_block) in enumerate(zip(decode_layer_up_list, decode_block_list)):
            skip = skip_connect_list[decode_idx + 1]
            decode_feature = decode_layer_up(decode_feature, skip, *args)
            decode_feature = decode_block(decode_feature, *args)
        decode_feature = decode_final_conv(decode_feature)
        return decode_feature
    
    def get_common_kwarg_dict(self, use_checkpoint=None, is_diffusion=False):
        if is_diffusion:
            emb_dim_list = self.emb_dim_list
            emb_type_list = self.emb_type_list
        else:
            emb_dim_list = self.emb_dim_list[-1:]
            emb_type_list = self.emb_type_list[-1:]
        common_kwarg_dict = {
            "norm": self.norm,
            "act": self.act,
            "dropout_proba": self.drop_prob,
            "emb_dim_list": emb_dim_list,
            "emb_type_list": emb_type_list,
            "img_dim": self.img_dim
        }
        if use_checkpoint is not None:
            common_kwarg_dict["use_checkpoint"] = use_checkpoint
        return common_kwarg_dict
    
    def set_encoder(self):
        block_size = self.block_size
        emb_dim_list = self.emb_dim_list
        emb_type_list = self.emb_type_list
        block_depth_list = self.block_depth_list
        norm = self.norm
        act = self.act
        use_checkpoint = self.use_checkpoint
        feature_channel = self.feature_channel
        conv_block = self.conv_block
        # Stem block
        self.stem = self.get_encode_stem()
        # Mixed 5b (Inception-A block):
        self.mixed_5b, self.mixed_5b_attn = self.get_encode_mixed_5b()
        # 10x block35 (Inception-ResNet-A block):
        self.block_35_mixed, self.block_35_up = self.get_inception_block(block_depth_list[0],
                                                                        self.get_inception_block_35)
        # Mixed 6a (Reduction-A block)
        self.mixed_6a, self.mixed_6a_attn = self.get_encode_mixed_6a()
        # 20x block17 (Inception-ResNet-B block)
        self.block_17_mixed, self.block_17_up = self.get_inception_block(block_depth_list[1],
                                                                        self.get_inception_block_17)
        # Mixed 7a (Reduction-B block)
        self.mixed_7a, self.mixed_7a_attn = self.get_encode_mixed_7a()
        # 10x block8 (Inception-ResNet-C block)
        self.block_8_mixed, self.block_8_up = self.get_inception_block(block_depth_list[2],
                                                                        self.get_inception_block_8)
        layer_idx = -1
        attn_info = self.get_attn_info(self.attn_info_list[layer_idx], self.num_head_list[layer_idx])
        self.encode_final_block = conv_block(block_size * 130, feature_channel, 3,
                                      norm=norm, act=act, emb_dim_list=emb_dim_list, emb_type_list=emb_type_list, attn_info=attn_info,
                                      use_checkpoint=use_checkpoint[layer_idx], image_shape=self.get_image_shape(5), img_dim=self.img_dim)
    
    
    def get_decoder(self, decode_out_channel, decode_out_act, decode_fn_str_list, is_diffusion=False):
        block_size = self.block_size
        decode_init_channel = self.decode_init_channel
        attn_info_list = self.attn_info_list
        num_head_list = self.num_head_list
        use_checkpoint = self.use_checkpoint

        decoder_layer_up_list = []
        decoder_block_list = []
        for decode_idx, skip_channel in enumerate(range(self.model_depth)):
            attn_info = self.get_attn_info(attn_info_list[4 - decode_idx], num_head_list[4 - decode_idx])
            common_kwarg_dict = self.get_common_kwarg_dict(use_checkpoint=use_checkpoint[4 - decode_idx], is_diffusion=is_diffusion)
            common_kwarg_dict["image_shape"] = self.get_image_shape(5 - decode_idx)
            decode_block_out_channel = decode_init_channel // (2 ** (decode_idx + 1))

            if decode_idx == 0:
                decode_in_channel = self.feature_channel
            else:
                decode_in_channel = decode_init_channel // (2 ** decode_idx)

            skip_channel = self.skip_channel_list[-(decode_idx + 2)]
            if self.is_encoder_unet() and decode_idx < 3:
                skip_channel += decode_init_channel // (2 ** 2) // (2 ** decode_idx)

            decoder_layer_up = MultiDecoderND_V3(decode_in_channel, skip_channel, decode_block_out_channel,
                                                kernel_size=2, decode_fn_str_list=decode_fn_str_list,
                                                use_residual_conv=self.use_residual_conv, attn_info=attn_info, **common_kwarg_dict)
            decoder_block = self.conv_block(decode_block_out_channel, decode_block_out_channel,
                                            kernel_size=3, stride=1, **common_kwarg_dict)
            decoder_layer_up_list.append(decoder_layer_up)
            decoder_block_list.append(decoder_block)
        decoder_layer_up_list = nn.ModuleList(decoder_layer_up_list)
        decoder_block_list = nn.ModuleList(decoder_block_list)
        decode_final_conv = OutputND(decode_block_out_channel, decode_out_channel,
                                     act=decode_out_act, img_dim=self.img_dim)
        return nn.ModuleList([decoder_layer_up_list, decoder_block_list, decode_final_conv])

    def get_validity_block(self, validity_shape, validity_act):
        validity_init_channel = self.block_size * 32
        common_kwarg_dict = self.get_common_kwarg_dict(use_checkpoint=self.use_checkpoint[4], is_diffusion=False)
        common_kwarg_dict["kernel_size"] = 3
        common_kwarg_dict["padding"] = 1
        common_kwarg_dict["image_shape"] = self.get_image_shape(5)
        validity_conv_1 = ResNetBlockND(self.feature_channel, validity_init_channel,
                                        **common_kwarg_dict)
        validity_conv_2 = ResNetBlockND(validity_init_channel,
                                            validity_init_channel // 2,
                                            **common_kwarg_dict)
        validity_conv_3 = ResNetBlockND(validity_init_channel // 2,
                                            validity_init_channel // 2,
                                            **common_kwarg_dict)
        gap_layer = None
        if self.img_dim == 1:
            gap_layer = nn.AdaptiveAvgPool1d
        elif self.img_dim == 2:
            gap_layer = nn.AdaptiveAvgPool2d
        elif self.img_dim == 3:
            gap_layer = nn.AdaptiveAvgPool3d
        validity_avg_pool = gap_layer(validity_shape[1:])
        validity_final_conv = ConvBlockND(validity_init_channel // 2, validity_shape[0],
                                                kernel_size=1, act=validity_act, norm=None, dropout_proba=0.0)
        validity_block = nn.Sequential(
            validity_conv_1,
            validity_conv_2,
            validity_conv_3,
            validity_avg_pool,
            validity_final_conv,
        )
        return validity_block
    
    def get_attn_info(self, attn_info, num_heads=None, dim_head=32):
        if attn_info is None:
            return None
        elif attn_info is False:
            return {"num_heads": num_heads, "dim_head": dim_head, "full_attn": False}
        elif attn_info is True:
            return {"num_heads": num_heads, "dim_head": dim_head, "full_attn": True}
    
    def get_attn_layer(self, dim, num_heads, dim_head, use_full_attn, use_checkpoint):
        if use_full_attn is True:
            return AttentionBlock(channels=dim, num_heads=num_heads,
                                  use_checkpoint=use_checkpoint)
        elif use_full_attn is False:
            return LinearAttention(dim=dim, num_heads=num_heads, dim_head=dim_head,
                                   use_checkpoint=use_checkpoint)
        else:
            return nn.Identity()

    def get_image_init_shape(self):
        image_shape = None
        if (self.norm == nn.RMSNorm) or (self.norm == nn.LayerNorm):
            image_shape = np.array(tuple(self.img_size for _ in range(self.img_dim)))
        return image_shape
    
    def get_image_shape(self, down_level):
        if self.image_shape is None:
            image_shape = None
        else:
            image_shape = self.image_shape // (2 ** down_level)
        return image_shape
    
    def is_encoder_unet(self):
        return getattr(self, "encoder_unet", False)
    
    def get_encode_stem(self):
        in_channel = self.in_channel
        block_size = self.block_size
        get_attn_info = self.get_attn_info
        attn_info_list = self.attn_info_list
        emb_type_list = self.emb_type_list
        common_kwarg_dict = self.get_common_kwarg_dict(use_checkpoint=None, is_diffusion=True)
        conv_block = self.conv_block

        stem_layer_0_1_init_channel = block_size * 8
        stem_layer_5_init_channel = block_size * 8
        if self.is_encoder_unet():
            stem_layer_0_1_init_channel += self.decode_init_channel // (2 ** 5)
            stem_layer_5_init_channel += self.decode_init_channel // (2 ** 4)
        return nn.ModuleDict({
            'stem_layer_0_0': conv_block(in_channel, block_size * 8, 3, stride=1,
                                        attn_info=get_attn_info(emb_type_list, attn_info_list[0], self.attn_dim_head[0]),
                                        use_checkpoint=self.use_checkpoint[0], image_shape=self.get_image_shape(0), **common_kwarg_dict),
            'stem_layer_0_1': conv_block(stem_layer_0_1_init_channel, block_size * 8, 3, stride=1,
                                        attn_info=get_attn_info(emb_type_list, attn_info_list[0], self.attn_dim_head[0]),
                                        use_checkpoint=self.use_checkpoint[0], image_shape=self.get_image_shape(0), **common_kwarg_dict),
            'stem_layer_1_0': conv_block(block_size * 8, block_size * 8, 3, stride=1,
                                        attn_info=get_attn_info(emb_type_list, attn_info_list[0], self.attn_dim_head[0]),
                                        use_checkpoint=self.use_checkpoint[0], image_shape=self.get_image_shape(0), **common_kwarg_dict),
            'stem_layer_1_1': conv_block(block_size * 8, block_size * 8, 3, stride=2, padding=self.padding_3x3,
                                        attn_info=get_attn_info(emb_type_list, attn_info_list[0], self.attn_dim_head[0]),
                                        use_checkpoint=self.use_checkpoint[0], image_shape=self.get_image_shape(0), **common_kwarg_dict),
            'stem_layer_2': conv_block(block_size * 8, block_size * 8, 3,
                                        attn_info=get_attn_info(emb_type_list, attn_info_list[0], self.attn_dim_head[0]),
                                        use_checkpoint=self.use_checkpoint[0], image_shape=self.get_image_shape(1), **common_kwarg_dict),
            'stem_layer_3': conv_block(block_size * 8, block_size * 8, 3,
                                        attn_info=get_attn_info(emb_type_list, attn_info_list[1], self.attn_dim_head[1]),
                                        use_checkpoint=self.use_checkpoint[1], image_shape=self.get_image_shape(1), **common_kwarg_dict),
            'stem_layer_4': get_maxpool_nd(self.img_dim)(3, stride=2, padding=self.padding_3x3),
            'stem_layer_5': conv_block(stem_layer_5_init_channel, block_size * 8, 1,
                                        attn_info=get_attn_info(emb_type_list, attn_info_list[1], self.attn_dim_head[1]),
                                        use_checkpoint=self.use_checkpoint[1], image_shape=self.get_image_shape(2), **common_kwarg_dict),
            'stem_layer_6': conv_block(block_size * 8, block_size * 12, 3,
                                        attn_info=get_attn_info(emb_type_list, attn_info_list[2], self.attn_dim_head[2]),
                                        use_checkpoint=self.use_checkpoint[2], image_shape=self.get_image_shape(2), **common_kwarg_dict),
            'stem_layer_7': get_maxpool_nd(self.img_dim)(3, stride=2, padding=self.padding_3x3)
        })
    
    def get_encode_mixed_5b(self):
        block_size = self.block_size
        layer_idx = 3
        common_kwarg_dict = self.get_common_kwarg_dict(use_checkpoint=self.use_checkpoint[layer_idx], is_diffusion=True)
        common_kwarg_dict["image_shape"] = self.get_image_shape(3)
        
        init_block_size = block_size * 12
        if self.is_encoder_unet():
            init_block_size += self.decode_init_channel // (2 ** 3)
        mixed_5b_branch_0_0 = ConvBlockND(init_block_size, block_size * 6, 1, **common_kwarg_dict)
        mixed_5b_branch_1_0 = ConvBlockND(init_block_size, block_size * 3, 1, **common_kwarg_dict)
        mixed_5b_branch_1_1 = ConvBlockND(block_size * 3, block_size * 4, 5, **common_kwarg_dict)
        mixed_5b_branch_2_0 = ConvBlockND(init_block_size, block_size * 4, 1, **common_kwarg_dict)
        mixed_5b_branch_2_1 = ConvBlockND(block_size * 4, block_size * 6, 3, **common_kwarg_dict)
        mixed_5b_branch_2_2 = ConvBlockND(block_size * 6, block_size * 6, 3, **common_kwarg_dict)
        mixed_5b_branch_pool_0 = get_avgpool_nd(self.img_dim)(3, stride=1, padding=1)
        mixed_5b_branch_pool_1 = ConvBlockND(init_block_size, block_size * 4, 1, **common_kwarg_dict)
        mixed_5b = nn.ModuleDict({
            "mixed_5b_branch_0": nn.ModuleList([mixed_5b_branch_0_0]),
            "mixed_5b_branch_1": nn.ModuleList([mixed_5b_branch_1_0, 
                                                mixed_5b_branch_1_1]),
            "mixed_5b_branch_2_0": nn.ModuleList([mixed_5b_branch_2_0,
                                                  mixed_5b_branch_2_1,
                                                  mixed_5b_branch_2_2]),
            "mixed_5b_branch_pool": nn.ModuleList([mixed_5b_branch_pool_0,
                                                   mixed_5b_branch_pool_1]),
        })
        if self.attn_info_list[layer_idx] is not None:
            attn_info = self.get_attn_info(self.attn_info_list[layer_idx], self.num_head_list[layer_idx])
            attn_layer = ConvBlockND(block_size * 20, block_size * 20, 1, attn_info=attn_info, **common_kwarg_dict)
        else:
            attn_layer = nn.Identity()
            
        return mixed_5b, attn_layer
    
    def get_encode_mixed_6a(self):
        padding_3x3 = self.padding_3x3
        block_size = self.block_size
        layer_idx = 3
        common_kwarg_dict = self.get_common_kwarg_dict(use_checkpoint=self.use_checkpoint[layer_idx], is_diffusion=True)
        common_kwarg_dict["image_shape"] = self.get_image_shape(3)
        init_block_size = block_size * 20
        if self.is_encoder_unet():
            init_block_size += self.decode_init_channel // (2 ** 2)

        mixed_6a_branch_0_0 = ConvBlockND(init_block_size, block_size * 24, 3,
                                          stride=2, padding=padding_3x3, **common_kwarg_dict)
        mixed_6a_branch_1_1 = ConvBlockND(init_block_size, block_size * 16, 1, **common_kwarg_dict)
        mixed_6a_branch_1_2 = ConvBlockND(block_size * 16, block_size * 16, 3, **common_kwarg_dict)
        mixed_6a_branch_1_3 = ConvBlockND(block_size * 16, block_size * 24, 3,
                                          stride=2, padding=padding_3x3, **common_kwarg_dict)
        mixed_6a_branch_pool_0 = get_maxpool_nd(self.img_dim)(3, stride=2, padding=padding_3x3)
        mixed_6a = nn.ModuleDict({
            "mixed_6a_branch_0": nn.ModuleList([mixed_6a_branch_0_0]),
            "mixed_6a_branch_1": nn.ModuleList([mixed_6a_branch_1_1,
                                                mixed_6a_branch_1_2,
                                                mixed_6a_branch_1_3]),
            "mixed_6a_branch_pool": nn.ModuleList([mixed_6a_branch_pool_0]),
        })
        if self.attn_info_list[layer_idx] is not None:
            attn_info = self.get_attn_info(self.attn_info_list[layer_idx], self.num_head_list[layer_idx])
            attn_layer = ConvBlockND(block_size * 68, block_size * 68, 1, attn_info=attn_info, **common_kwarg_dict)
        else:
            attn_layer = nn.Identity()
        return mixed_6a, attn_layer
    
    def get_encode_mixed_7a(self):
        padding_3x3 = self.padding_3x3
        block_size = self.block_size
        layer_idx = 4
        common_kwarg_dict = self.get_common_kwarg_dict(use_checkpoint=self.use_checkpoint[layer_idx], is_diffusion=True)
        common_kwarg_dict["image_shape"] = self.get_image_shape(4)
        init_block_size = block_size * 68
        attn_block_size = block_size * 130
        anch_block_size = self.decode_init_channel // (2 ** 2) + self.decode_init_channel // (2 ** 1)
        if self.is_encoder_unet():
            init_block_size += anch_block_size
            attn_block_size += anch_block_size
        mixed_7a_branch_0_1 = ConvBlockND(init_block_size, block_size * 16, 1, **common_kwarg_dict)
        mixed_7a_branch_0_2 = ConvBlockND(block_size * 16, block_size * 24, 3,
                                          stride=2, padding=padding_3x3, **common_kwarg_dict)
        mixed_7a_branch_1_1 = ConvBlockND(init_block_size, block_size * 16, 1, **common_kwarg_dict)
        mixed_7a_branch_1_2 = ConvBlockND(block_size * 16, block_size * 18, 3,
                                          stride=2, padding=padding_3x3, **common_kwarg_dict)
        mixed_7a_branch_2_1 = ConvBlockND(init_block_size, block_size * 16, 1, **common_kwarg_dict)
        mixed_7a_branch_2_2 = ConvBlockND(block_size * 16, block_size * 18, 3, **common_kwarg_dict)
        mixed_7a_branch_2_3 = ConvBlockND(block_size * 18, block_size * 20, 3,
                                          stride=2, padding=padding_3x3, **common_kwarg_dict)
        mixed_7a_branch_pool_0 = get_maxpool_nd(self.img_dim)(3, stride=2, padding=padding_3x3)
        mixed_7a = nn.ModuleDict({
            "mixed_7a_branch_0": nn.ModuleList([mixed_7a_branch_0_1,
                                                mixed_7a_branch_0_2]),
            "mixed_7a_branch_1": nn.ModuleList([mixed_7a_branch_1_1,
                                                mixed_7a_branch_1_2]),
            "mixed_7a_branch_2_1": nn.ModuleList([mixed_7a_branch_2_1,
                                                  mixed_7a_branch_2_2,
                                                  mixed_7a_branch_2_3]),
            "mixed_7a_branch_pool": nn.ModuleList([mixed_7a_branch_pool_0]),
        })
        if self.attn_info_list[layer_idx] is not None:
            attn_info = self.get_attn_info(self.attn_info_list[layer_idx], self.num_head_list[layer_idx])
            attn_layer = ConvBlockND(attn_block_size, block_size * 130, 1, attn_info=attn_info, **common_kwarg_dict)
        else:
            attn_layer = nn.Identity()
        return mixed_7a, attn_layer
    
    def get_inception_block(self, block_depth, get_block_fn):
        block_list = [get_block_fn()
                      for idx in range(block_depth)]
        mixed_list, up_list = [list(item) for item in zip(*block_list)]
        return nn.ModuleList(mixed_list), nn.ModuleList(up_list)

    def get_inception_block_35(self):

        block_size = self.block_size
        in_channels = block_size * 20
        mixed_channel = block_size * 8
        layer_idx = 3
        common_kwarg_dict = self.get_common_kwarg_dict(use_checkpoint=self.use_checkpoint[layer_idx], is_diffusion=True)
        common_kwarg_dict["image_shape"] = self.get_image_shape(3)
        branch_0_0 = ConvBlockND(in_channels, block_size * 2, 1, **common_kwarg_dict)
        branch_1_0 = ConvBlockND(in_channels, block_size * 2, 1, **common_kwarg_dict)
        branch_1_1 = ConvBlockND(block_size * 2, block_size * 2, 3, **common_kwarg_dict)
        branch_2_0 = ConvBlockND(in_channels, block_size * 2, 1, **common_kwarg_dict)
        branch_2_1 = ConvBlockND(block_size * 2, block_size * 3, 3, **common_kwarg_dict)
        branch_2_2 = ConvBlockND(block_size * 3, block_size * 4, 3, **common_kwarg_dict)

        up = ConvBlockND(mixed_channel, in_channels, 1,
                        bias=True, norm=self.norm, act=None,
                        emb_dim_list=self.emb_dim_list, emb_type_list=self.emb_type_list,
                        use_checkpoint=self.use_checkpoint[layer_idx], image_shape=self.get_image_shape(4), img_dim=self.img_dim)
        mixed = nn.ModuleDict({
            "branch_0": nn.ModuleList([branch_0_0]),
            "branch_1": nn.ModuleList([branch_1_0,
                                    branch_1_1]),
            "branch_2_1": nn.ModuleList([branch_2_0,
                                        branch_2_1,
                                        branch_2_2])
        })
        return mixed, up

    def get_inception_block_17(self):

        block_size = self.block_size
        in_channels = block_size * 68
        if self.is_encoder_unet():
            in_channels += self.decode_init_channel // (2 ** 2)
        mixed_channel = block_size * 24
        layer_idx = 3
        common_kwarg_dict = self.get_common_kwarg_dict(use_checkpoint=self.use_checkpoint[layer_idx], is_diffusion=True)
        common_kwarg_dict["image_shape"] = self.get_image_shape(4)
        branch_0_0 = ConvBlockND(in_channels, block_size * 12, 1, **common_kwarg_dict)
        branch_1_0 = ConvBlockND(in_channels, block_size * 8, 1, **common_kwarg_dict)
        branch_1_list = [branch_1_0]
        if self.img_dim == 1:
            branch_1_1 = ConvBlockND(block_size * 8, block_size * 12, 7, **common_kwarg_dict)
            branch_1_list.append(branch_1_1)
        elif self.img_dim == 2:
            branch_1_1 = ConvBlockND(block_size * 8, block_size * 10, [1, 7], **common_kwarg_dict)
            branch_1_2 = ConvBlockND(block_size * 10, block_size * 12, [7, 1], **common_kwarg_dict)
            branch_1_list.extend([branch_1_1, branch_1_2])
        elif self.img_dim == 3:
            branch_1_1 = ConvBlockND(block_size * 8, block_size * 10, [1, 1, 7], **common_kwarg_dict)
            branch_1_2 = ConvBlockND(block_size * 10, block_size * 10, [1, 7, 1], **common_kwarg_dict)
            branch_1_3 = ConvBlockND(block_size * 10, block_size * 12, [7, 1, 1], **common_kwarg_dict)
            branch_1_list.extend([branch_1_1, branch_1_2, branch_1_3])

        up = ConvBlockND(mixed_channel, in_channels, 1,
                        bias=True, norm=self.norm, act=None,
                        emb_dim_list=self.emb_dim_list, emb_type_list=self.emb_type_list,
                        use_checkpoint=self.use_checkpoint[layer_idx], img_dim=self.img_dim)
        mixed = nn.ModuleDict({
            "branch_0": nn.ModuleList([branch_0_0]),
            "branch_1": nn.ModuleList(branch_1_list),
        })
        return mixed, up

    def get_inception_block_8(self):
        block_size = self.block_size

        in_channels = block_size * 130
        mixed_channel = block_size * 28
        layer_idx = 4
        common_kwarg_dict = self.get_common_kwarg_dict(use_checkpoint=self.use_checkpoint[layer_idx], is_diffusion=True)
        common_kwarg_dict["image_shape"] = self.get_image_shape(5)
        branch_0_0 = ConvBlockND(in_channels, block_size * 12, 1, **common_kwarg_dict)
        branch_1_0 = ConvBlockND(in_channels, block_size * 12, 1, **common_kwarg_dict)
        branch_1_list = [branch_1_0]
        if self.img_dim == 1:
            branch_1_1 = ConvBlockND(block_size * 12, block_size * 16, 3, **common_kwarg_dict)
            branch_1_list.append(branch_1_1)
        elif self.img_dim == 2:
            branch_1_1 = ConvBlockND(block_size * 12, block_size * 14, [1, 3], **common_kwarg_dict)
            branch_1_2 = ConvBlockND(block_size * 14, block_size * 16, [3, 1], **common_kwarg_dict)
            branch_1_list.extend([branch_1_1, branch_1_2])
        elif self.img_dim == 3:
            branch_1_1 = ConvBlockND(block_size * 12, block_size * 14, [1, 1, 3], **common_kwarg_dict)
            branch_1_2 = ConvBlockND(block_size * 14, block_size * 14, [1, 3, 1], **common_kwarg_dict)
            branch_1_3 = ConvBlockND(block_size * 14, block_size * 16, [3, 1, 1], **common_kwarg_dict)
            branch_1_list.extend([branch_1_1, branch_1_2, branch_1_3])
        up = ConvBlockND(mixed_channel, in_channels, 1,
                        bias=True, norm=self.norm, act=None,
                        emb_dim_list=self.emb_dim_list, emb_type_list=self.emb_type_list,
                        use_checkpoint=self.use_checkpoint[layer_idx], img_dim=self.img_dim)
        
        mixed = nn.ModuleDict({
            "branch_0": nn.ModuleList([branch_0_0]),
            "branch_1": nn.ModuleList(branch_1_list),
        })
        return mixed, up
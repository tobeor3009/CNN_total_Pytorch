from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.nn import init
from .nn import timestep_embedding
from .diffusion_layer import default, Return
from .diffusion_layer import ConvBlockND, ResNetBlockND, ResNetBlockNDSkip, MultiDecoderND_V2, OutputND
from .diffusion_layer import get_maxpool_nd, get_avgpool_nd
from .diffusion_layer import LinearAttention, Attention, AttentionBlock, MaxPool2d, AvgPool2d, MultiInputSequential
from .diffusion_layer import default, prob_mask_like, LearnedSinusoidalPosEmb, SinusoidalPosEmb, GroupNorm32
from .diffusion_layer import feature_z_normalize, z_normalize
from .diffusion_layer import ClassificationHeadSimple
from src.model.inception_resnet_v2.common_module.layers import get_act, get_norm
from einops import rearrange, repeat

def get_skip_connect_channel_list(block_size, mini=False):
    if mini:
        return np.array([block_size * 2, block_size * 4, block_size * 12])
    else:
        return np.array([block_size * 2, block_size * 2, block_size * 4, block_size * 12,
                        block_size * 68])

def get_encode_feature_channel(block_size, last_channel_ratio):
    feature_channel = block_size * 96 * last_channel_ratio
    return int(round(feature_channel))

def get_time_emb_dim(block_size):
    # emb_dim = block_size * 16
    # I found 512 is best size for all size
    emb_dim = 512
    time_emb_dim_init = emb_dim // 2
    time_emb_dim = emb_dim * 4
    return emb_dim, time_emb_dim_init, time_emb_dim

class InceptionResNetV2_UNet(nn.Module):
    def __init__(self, in_channel=3, cond_channel=3, img_size=256, block_size=8,
                 emb_channel=1024, decode_init_channel=None, block_depth_info="mini",
                 norm=GroupNorm32, act="silu", num_class_embeds=None, drop_prob=0.05, cond_drop_prob=0.5,
                 self_condition=False, use_checkpoint=[False, False, False, False, True],
                 attn_info_list=[None, None, None, None, True], attn_dim_head=32, num_head_list=[1, 1, 1, 1, 1],
                 diffusion_out_channel=1, diffusion_act=None, diffusion_decode_fn_str_list=["upsample"],
                 seg_out_channel=2, seg_act="softmax", seg_decode_fn_str_list=["conv_transpose"],
                 class_out_channel=2, class_act="softmax",
                 recon_out_channel=None, recon_act="tanh", recon_decode_fn_str_list=["conv_transpose"],
                 validity_shape=(1, 8, 8), validity_act=None,
                 get_diffusion=False, get_seg=True, get_class=False, get_recon=False, get_validity=False,
                 include_encoder=False, include_latent_net=False, img_dim=2, encoder_img_dim=None):
        super().__init__()
        last_channel_ratio = 1
        self.include_encoder = include_encoder
        self.include_latent_net = include_latent_net
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

        if norm == nn.RMSNorm:
            self.image_shape = np.array(tuple(img_size for _ in range(img_dim)))
        else:
            self.image_shape = None

        # for compability with diffusion_sample
        self.img_size = img_size
        self.cond_channel = cond_channel
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
        self.img_dim = img_dim
        self.encoder_img_dim = encoder_img_dim or img_dim
        ##################################
        assert len(attn_info_list) == 5, "check len(attn_info_list) == 5"
        assert len(use_checkpoint) == 5, "check len(use_checkpoint) == 5"
        assert len(num_head_list) == 5, "check len(num_head_list) == 5"
        assert isinstance(img_dim, int), "img_dim must be int"
        ##################################
        self.use_non_diffusion = get_seg or get_class or get_recon or get_validity
        self.get_diffusion = get_diffusion
        self.get_seg = get_seg
        self.get_class = get_class
        self.get_recon = get_recon
        self.get_validity = get_validity
        ##################################
        emb_dim, time_emb_dim_init, time_emb_dim = get_time_emb_dim(block_size)
        self.time_emb_dim_init = time_emb_dim_init
        emb_dim_list = []
        emb_type_list = []
        if get_diffusion:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim_init, time_emb_dim),
                get_act(act),
                nn.Linear(time_emb_dim, time_emb_dim)
            )
            emb_dim_list.append(time_emb_dim)
            emb_type_list.append("seq")
            if self.include_encoder:
                if isinstance(include_encoder, nn.Module):
                    self.encoder = include_encoder
                else:
                    self.encoder = InceptionResNetV2_Encoder(in_channel=cond_channel, img_size=img_size, block_size=block_size,
                                                                emb_channel=emb_channel, norm=norm, act=act, last_channel_ratio=last_channel_ratio,
                                                                use_checkpoint=use_checkpoint, attn_info_list=attn_info_list,
                                                                attn_dim_head=attn_dim_head, num_head_list=num_head_list,
                                                                block_depth_info=block_depth_info, img_dim=self.encoder_img_dim)
                emb_dim_list.append(emb_channel)
                emb_type_list.append("cond")
            if self.include_latent_net:
                if isinstance(include_latent_net, nn.Module):
                    self.latent_net = include_latent_net
                else:
                    self.emb_channel = emb_channel
                    self.latent_net = MLPSkipNet(emb_channel=emb_channel, block_size=block_size, num_time_layers=2, time_last_act=None,
                                                num_latent_layers=10, latent_last_act=None, latent_dropout=0., latent_condition_bias=1,
                                                act=act, use_norm=True, skip_layers=[1, 2, 3, 4, 5, 6, 7, 8, 9])
        
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
        if get_diffusion:
            self.diffusion_decoder_list = self.get_decoder(diffusion_out_channel, diffusion_act,
                                                           decode_fn_str_list=diffusion_decode_fn_str_list, is_diffusion=True)
        if get_seg:
            self.seg_decoder_list = self.get_decoder(seg_out_channel, seg_act,
                                                     decode_fn_str_list=seg_decode_fn_str_list, is_diffusion=False)
        if get_class:
            self.class_head = ClassificationHeadSimple(self.feature_channel,
                                                      class_out_channel, drop_prob, class_act, img_dim)
        if get_recon:
            recon_out_channel = recon_out_channel or in_channel
            self.recon_decoder_list = self.get_decoder(recon_out_channel, recon_act,
                                                       decode_fn_str_list=recon_decode_fn_str_list, is_diffusion=False)
        if get_validity:
            self.validity_head = self.get_validity_block(validity_shape, validity_act)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x, t=None, t_cond=None,
                x_start=None, cond=None,
                x_self_cond=None, class_labels=None, cond_drop_prob=None,
                infer_diffusion=True):
        output = None
        class_emb = self.process_class_emb(x, class_labels, cond_drop_prob)
        
        if infer_diffusion:
            output = self._forward_diffusion(x=x, t=t, t_cond=t_cond, x_start=x_start, cond=cond,
                                             x_self_cond=x_self_cond, class_emb=class_emb)
        else:
            output = self._forward_non_diffusion(x=x, class_emb=class_emb)
        return output
    
    def _forward_diffusion(self, x, t, t_cond=None,
                            x_start=None, cond=None,
                            x_self_cond=None, class_emb=None):
        output = Return()
        emb_list = []
        # autoencoder original source not used t_cond varaiable
        if t_cond is None:
            t_cond = t
        time_emb = timestep_embedding(t, self.time_emb_dim_init)
        time_emb = self.time_mlp(time_emb)
        emb_list.append(time_emb)
        if cond is None and self.include_encoder:
            assert x.shape == x_start.shape, f"x.shape: {x.shape}, x_start.shape: {x_start.shape}"
            latent_feature = self.encoder(x_start)
        else:
            latent_feature = cond
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
    
    def _forward_non_diffusion(self, x, class_emb):
        output = Return()
        non_diffusion_emb_list = []

        if self.include_encoder:
            latent_feature = self.encoder(x)
            non_diffusion_emb_list.append(None)
            non_diffusion_emb_list.append(latent_feature)
        if class_emb is None:
            non_diffusion_emb_list.append(class_emb)

        encode_feature, skip_connect_list = self.encode_forward(x, *non_diffusion_emb_list)
        if self.get_seg:
            seg_decode_feature = self.decode_forward(self.seg_decoder_list, encode_feature, skip_connect_list, *non_diffusion_emb_list)
            output["seg_pred"] = seg_decode_feature
        if self.get_class:
            class_output = self.class_head(x)
            output["class_pred"] = class_output
        if self.get_recon:
            recon_decode_feature = self.decode_forward(self.recon_decoder_list, encode_feature, skip_connect_list, *non_diffusion_emb_list)
            output["recon_pred"] = recon_decode_feature
        if self.get_validity:
            validitiy_output = self.validity_head(x)
            output["validitiy_pred"] = validitiy_output
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
        block_35 = self.process_inception_block(self.block_35_mixed, self.block_35_attn, self.block_35_up,
                                                mixed_5b, self.block_scale_list[0], *args)
        # mixed_6a: skip connect target
        mixed_6a = self.process_encode_block(self.mixed_6a, self.mixed_6a_attn, block_35, *args)
        skip_connect_list.append(mixed_6a)
        # block_17
        block_17 = self.process_inception_block(self.block_17_mixed, self.block_17_attn, self.block_17_up,
                                                mixed_6a, self.block_scale_list[1], *args)
        # mixed_7a: skip connect target
        mixed_7a = self.process_encode_block(self.mixed_7a, self.mixed_7a_attn, block_17, *args)
        # skip_connect_list.append(mixed_7a)
        # block_8
        block_8 = self.process_inception_block(self.block_8_mixed, self.block_8_attn, self.block_8_up,
                                               mixed_7a, self.block_scale_list[2], *args)
        # final_output
        output = self.encode_final_block1(block_8, *args)
        output = self.encode_final_attn(output)
        output = self.encode_final_block2(output, *args)

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

    def process_inception_block(self, block_mixed, block_attn, block_up,
                                x, scale, *args):
        for mixed, up_layer in zip(block_mixed, block_up):
            x_temp = self.process_encode_block(mixed, None, x, *args)
            x_temp = up_layer(x_temp, *args)
            x = x + x_temp * scale
            x = self.act_layer(x)
        x = block_attn(x)
        return x
        
    def decode_forward(self, decoder_list, encode_feature, skip_connect_list, *args):
        decode_block_list, decode_layer_up_list, decode_final_conv = decoder_list
        decode_feature = encode_feature
        for decode_idx, (decode_block, decode_layer_up) in enumerate(zip(decode_block_list, decode_layer_up_list)):
            if decode_idx > 0:
                skip = skip_connect_list[decode_idx - 1]
                decode_feature = decode_block(decode_feature, skip, *args)
            else:
                decode_feature = decode_block(decode_feature, *args)
            decode_feature = decode_layer_up(decode_feature, *args)
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
        # Stem block
        self.stem = self.get_encode_stem()
        # Mixed 5b (Inception-A block):
        self.mixed_5b, self.mixed_5b_attn = self.get_encode_mixed_5b()
        # 10x block35 (Inception-ResNet-A block):
        self.block_35_mixed, self.block_35_attn, self.block_35_up = self.get_inception_block(block_depth_list[0],
                                                                                             self.get_inception_block_35,
                                                                                             block_size * 20, 3)
        # Mixed 6a (Reduction-A block)
        self.mixed_6a, self.mixed_6a_attn = self.get_encode_mixed_6a()
        # 20x block17 (Inception-ResNet-B block)
        self.block_17_mixed, self.block_17_attn, self.block_17_up = self.get_inception_block(block_depth_list[1],
                                                                                             self.get_inception_block_17,
                                                                                             block_size * 68, 3)
        # Mixed 7a (Reduction-B block)
        self.mixed_7a, self.mixed_7a_attn = self.get_encode_mixed_7a()
        # 10x block8 (Inception-ResNet-C block)
        self.block_8_mixed, self.block_8_attn, self.block_8_up = self.get_inception_block(block_depth_list[2],
                                                                                          self.get_inception_block_8,
                                                                                          block_size * 130, 4)
        layer_idx = 4
        # attn_info = self.get_attn_info(self.attn_info_list[layer_idx], self.num_head_list[layer_idx])
        self.encode_final_block1 = ResNetBlockND(block_size * 130, feature_channel, 3,
                                      norm=norm, act=act, emb_dim_list=emb_dim_list, emb_type_list=emb_type_list, attn_info=None,
                                      use_checkpoint=use_checkpoint[layer_idx], image_shape=self.get_image_shape(5), img_dim=self.img_dim)
        self.encode_final_attn = self.get_attn_layer(feature_channel, self.num_head_list[layer_idx], self.attn_dim_head[layer_idx],
                                                     True, use_checkpoint=self.use_checkpoint[layer_idx])
        self.encode_final_block2 = ResNetBlockND(feature_channel, feature_channel, 3,
                                                norm=norm, act=act, emb_dim_list=emb_dim_list, emb_type_list=emb_type_list, attn_info=None,
                                                use_checkpoint=use_checkpoint[layer_idx], image_shape=self.get_image_shape(5), img_dim=self.img_dim)
    
    
    def get_decoder(self, decode_out_channel, decode_out_act, decode_fn_str_list, is_diffusion=False):
        block_size = self.block_size
        decode_init_channel = self.decode_init_channel
        attn_info_list = self.attn_info_list
        num_head_list = self.num_head_list
        use_checkpoint = self.use_checkpoint

        decoder_block_list = []
        decoder_layer_up_list = []
        skip_channel_list = get_skip_connect_channel_list(self.block_size)
        for decode_idx, skip_channel in enumerate(skip_channel_list):
            attn_info = self.get_attn_info(attn_info_list[4 - decode_idx], num_head_list[4 - decode_idx])
            common_kwarg_dict = self.get_common_kwarg_dict(use_checkpoint=use_checkpoint[4 - decode_idx], is_diffusion=is_diffusion)
            common_kwarg_dict["attn_info"] = attn_info
            common_kwarg_dict["image_shape"] = self.get_image_shape(5 - decode_idx)
            decode_block_out_channel = decode_init_channel // (2 ** (decode_idx + 1))

            if decode_idx == 0:
                decode_in_channel = block_size * 96
            else:
                decode_in_channel = decode_init_channel // (2 ** decode_idx)


            if decode_idx > 0:
                skip_channel = skip_channel_list[4 - decode_idx + 1]
                decoder_block = ResNetBlockNDSkip(decode_in_channel + skip_channel, decode_block_out_channel,
                                                  kernel_size=3, stride=1, **common_kwarg_dict)
            else:
                decoder_block = ResNetBlockND(decode_in_channel, decode_block_out_channel, 
                                              kernel_size=3, stride=1, **common_kwarg_dict)

            decoder_layer_up = MultiDecoderND_V2(decode_block_out_channel, decode_block_out_channel,
                                                kernel_size=2, decode_fn_str_list=decode_fn_str_list, **common_kwarg_dict)
            decoder_block_list.append(decoder_block)
            decoder_layer_up_list.append(decoder_layer_up)
        decoder_block_list = nn.ModuleList(decoder_block_list)
        decoder_layer_up_list = nn.ModuleList(decoder_layer_up_list)
        decode_final_conv = OutputND(decode_block_out_channel, decode_out_channel,
                                     act=decode_out_act, img_dim=self.img_dim)
        return nn.ModuleList([decoder_block_list, decoder_layer_up_list, decode_final_conv])

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

    def get_image_shape(self, down_level):
        if self.image_shape is None:
            image_shape = None
        else:
            image_shape = self.image_shape // (2 ** down_level)
        return image_shape
    
    def get_encode_stem(self):
        in_channel = self.in_channel
        block_size = self.block_size
        get_attn_info = self.get_attn_info
        attn_info_list = self.attn_info_list
        emb_type_list = self.emb_type_list
        common_kwarg_dict = self.get_common_kwarg_dict(use_checkpoint=None, is_diffusion=True)
        return nn.ModuleDict({
            'stem_layer_0_0': ResNetBlockND(in_channel, block_size * 2, 3, stride=1,
                                        attn_info=get_attn_info(emb_type_list, attn_info_list[0], self.attn_dim_head[0]),
                                        use_checkpoint=self.use_checkpoint[0], image_shape=self.get_image_shape(0), **common_kwarg_dict),
            'stem_layer_0_1': ResNetBlockND(block_size * 2, block_size * 2, 3, stride=1,
                                        attn_info=get_attn_info(emb_type_list, attn_info_list[0], self.attn_dim_head[0]),
                                        use_checkpoint=self.use_checkpoint[0], image_shape=self.get_image_shape(0), **common_kwarg_dict),
            'stem_layer_1_0': ResNetBlockND(block_size * 2, block_size * 2, 3, stride=1,
                                        attn_info=get_attn_info(emb_type_list, attn_info_list[0], self.attn_dim_head[0]),
                                        use_checkpoint=self.use_checkpoint[0], image_shape=self.get_image_shape(0), **common_kwarg_dict),
            'stem_layer_1_1': ResNetBlockND(block_size * 2, block_size * 2, 3, stride=2, padding=self.padding_3x3,
                                        attn_info=get_attn_info(emb_type_list, attn_info_list[0], self.attn_dim_head[0]),
                                        use_checkpoint=self.use_checkpoint[0], image_shape=self.get_image_shape(0), **common_kwarg_dict),
            'stem_layer_2': ResNetBlockND(block_size * 2, block_size * 2, 3,
                                        attn_info=get_attn_info(emb_type_list, attn_info_list[0], self.attn_dim_head[0]),
                                        use_checkpoint=self.use_checkpoint[0], image_shape=self.get_image_shape(1), **common_kwarg_dict),
            'stem_layer_3': ResNetBlockND(block_size * 2, block_size * 4, 3,
                                        attn_info=get_attn_info(emb_type_list, attn_info_list[1], self.attn_dim_head[1]),
                                        use_checkpoint=self.use_checkpoint[1], image_shape=self.get_image_shape(1), **common_kwarg_dict),
            'stem_layer_4': get_maxpool_nd(self.img_dim)(3, stride=2, padding=self.padding_3x3),
            'stem_layer_5': ResNetBlockND(block_size * 4, block_size * 8, 1,
                                        attn_info=get_attn_info(emb_type_list, attn_info_list[1], self.attn_dim_head[1]),
                                        use_checkpoint=self.use_checkpoint[1], image_shape=self.get_image_shape(2), **common_kwarg_dict),
            'stem_layer_6': ResNetBlockND(block_size * 8, block_size * 12, 3,
                                        attn_info=get_attn_info(emb_type_list, attn_info_list[2], self.attn_dim_head[2]),
                                        use_checkpoint=self.use_checkpoint[2], image_shape=self.get_image_shape(2), **common_kwarg_dict),
            'stem_layer_7': get_maxpool_nd(self.img_dim)(3, stride=2, padding=self.padding_3x3)
        })
    
    def get_encode_mixed_5b(self):
        block_size = self.block_size
        layer_idx = 3
        common_kwarg_dict = self.get_common_kwarg_dict(use_checkpoint=self.use_checkpoint[layer_idx], is_diffusion=True)
        common_kwarg_dict["image_shape"] = self.get_image_shape(3)
        mixed_5b_branch_0_0 = ConvBlockND(block_size * 12, block_size * 6, 1, **common_kwarg_dict)
        mixed_5b_branch_1_0 = ConvBlockND(block_size * 12, block_size * 3, 1, **common_kwarg_dict)
        mixed_5b_branch_1_1 = ConvBlockND(block_size * 3, block_size * 4, 5, **common_kwarg_dict)
        mixed_5b_branch_2_0 = ConvBlockND(block_size * 12, block_size * 4, 1, **common_kwarg_dict)
        mixed_5b_branch_2_1 = ConvBlockND(block_size * 4, block_size * 6, 3, **common_kwarg_dict)
        mixed_5b_branch_2_2 = ConvBlockND(block_size * 6, block_size * 6, 3, **common_kwarg_dict)
        mixed_5b_branch_pool_0 = get_avgpool_nd(self.img_dim)(3, stride=1, padding=1)
        mixed_5b_branch_pool_1 = ConvBlockND(block_size * 12, block_size * 4, 1, **common_kwarg_dict)
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
        attn_layer = self.get_attn_layer(block_size * 20, self.num_head_list[layer_idx], self.attn_dim_head[layer_idx],
                                         self.attn_info_list[layer_idx], use_checkpoint=self.use_checkpoint[layer_idx])

        return mixed_5b, attn_layer
    
    def get_encode_mixed_6a(self):
        padding_3x3 = self.padding_3x3
        block_size = self.block_size
        layer_idx = 3
        common_kwarg_dict = self.get_common_kwarg_dict(use_checkpoint=self.use_checkpoint[layer_idx], is_diffusion=True)
        common_kwarg_dict["image_shape"] = self.get_image_shape(3)

        mixed_6a_branch_0_0 = ConvBlockND(block_size * 20, block_size * 24, 3,
                                          stride=2, padding=padding_3x3, **common_kwarg_dict)
        mixed_6a_branch_1_1 = ConvBlockND(block_size * 20, block_size * 16, 1, **common_kwarg_dict)
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
        attn_layer = self.get_attn_layer(block_size * 68, self.num_head_list[layer_idx], self.attn_dim_head[layer_idx],
                                         self.attn_info_list[layer_idx], use_checkpoint=self.use_checkpoint[layer_idx])
        return mixed_6a, attn_layer
    
    def get_encode_mixed_7a(self):
        padding_3x3 = self.padding_3x3
        block_size = self.block_size
        layer_idx = 4
        common_kwarg_dict = self.get_common_kwarg_dict(use_checkpoint=self.use_checkpoint[layer_idx], is_diffusion=True)
        common_kwarg_dict["image_shape"] = self.get_image_shape(4)

        mixed_7a_branch_0_1 = ConvBlockND(block_size * 68, block_size * 16, 1, **common_kwarg_dict)
        mixed_7a_branch_0_2 = ConvBlockND(block_size * 16, block_size * 24, 3,
                                          stride=2, padding=padding_3x3, **common_kwarg_dict)
        mixed_7a_branch_1_1 = ConvBlockND(block_size * 68, block_size * 16, 1, **common_kwarg_dict)
        mixed_7a_branch_1_2 = ConvBlockND(block_size * 16, block_size * 18, 3,
                                          stride=2, padding=padding_3x3, **common_kwarg_dict)
        mixed_7a_branch_2_1 = ConvBlockND(block_size * 68, block_size * 16, 1, **common_kwarg_dict)
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
        attn_layer = self.get_attn_layer(block_size * 130, self.num_head_list[layer_idx], self.attn_dim_head[layer_idx],
                                         self.attn_info_list[layer_idx], use_checkpoint=self.use_checkpoint[layer_idx])
        return mixed_7a, attn_layer
    
    def get_inception_block(self, block_depth, get_block_fn, out_channel, layer_idx):
        block_list = [get_block_fn() for _ in range(block_depth)]
        mixed_list, up_list = [list(item) for item in zip(*block_list)]
        attn_layer = self.get_attn_layer(out_channel, self.num_head_list[layer_idx], self.attn_dim_head[layer_idx],
                                         self.attn_info_list[layer_idx], use_checkpoint=self.use_checkpoint[layer_idx])
        return nn.ModuleList(mixed_list), attn_layer, nn.ModuleList(up_list)

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

class InceptionResNetV2_Encoder(InceptionResNetV2_UNet):
    def __init__(self, in_channel, img_size, block_size=16, emb_channel=1024, drop_prob=0.0,
                 norm=GroupNorm32, act="silu", last_channel_ratio=1,
                 use_checkpoint=False, attn_info_list=[None, False, False, False, True],
                 attn_dim_head=32, num_head_list=[2, 4, 8, 8, 16],
                 block_depth_info="mini", img_dim=2):
        super(InceptionResNetV2_UNet, self).__init__()

        self.use_inception_block_attn = True
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
        self.emb_dim_list = []
        self.emb_type_list = []
        self.num_head_list = num_head_list
        self.drop_prob = drop_prob
        self.block_scale_list = [0.17, 0.1, 0.2]
        self.act_layer = get_act(act)
        self.img_dim = img_dim
        self.feature_channel = get_encode_feature_channel(block_size, last_channel_ratio)

        if norm == nn.RMSNorm:
            self.image_shape = np.array(tuple(img_size for _ in range(img_dim)))
        else:
            self.image_shape = None

        ##################################
        self.set_encoder()
        image_shape = self.get_image_shape(5)
        if norm == nn.RMSNorm:
            norm_shape = (self.feature_channel, *image_shape)
        else:
            norm_shape = self.feature_channel
        self.pool_layer = nn.Sequential(
            self.norm(norm_shape),
            get_act(self.act),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(self.feature_channel, emb_channel)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        encode_feature, _ = super().encode_forward(x)
        encode_feature = self.pool_layer(encode_feature)
        return encode_feature

class MLPSkipNet(nn.Module):
    """
    concat x to hidden layers

    default MLP for the latent DPM in the paper!
    """
    def __init__(self, emb_channel, block_size=16, num_time_layers=2, time_last_act=None,
                 num_latent_layers=10, latent_last_act=None, latent_dropout=0., latent_condition_bias=1,
                 act="silu", use_norm=True, skip_layers=[1, 2, 3, 4, 5, 6, 7, 8, 9]):
        super().__init__()
        self.skip_layers = skip_layers
        _, time_emb_dim_init, time_emb_dim = get_time_emb_dim(block_size)
        self.time_emb_dim_init = time_emb_dim_init
        latent_hid_channels = time_emb_dim * 2
        layers = []
        for i in range(num_time_layers):
            if i == 0:
                a = time_emb_dim_init
                b = time_emb_dim
            else:
                a = time_emb_dim
                b = time_emb_dim
            layers.append(nn.Linear(a, b))
            if i < num_time_layers - 1:
                layers.append(get_act(time_last_act))
        self.time_embed = nn.Sequential(*layers)

        self.layers = nn.ModuleList([])
        for i in range(num_latent_layers):
            if i == 0:
                mlp_act = act
                mlp_norm = use_norm
                cond = True
                a, b = emb_channel, latent_hid_channels
                mlp_dropout = latent_dropout
            elif i == num_latent_layers - 1:
                mlp_act = None
                mlp_norm = False
                cond = False
                a, b = latent_hid_channels, emb_channel
                mlp_dropout = 0
            else:
                mlp_act = act
                mlp_norm = use_norm
                cond = True
                a, b = latent_hid_channels, latent_hid_channels
                mlp_dropout = latent_dropout

            if i in skip_layers:
                a += emb_channel

            self.layers.append(
                MLPLNAct(
                    a,
                    b,
                    norm=mlp_norm,
                    activation=mlp_act,
                    cond_channels=time_emb_dim,
                    use_cond=cond,
                    condition_bias=latent_condition_bias,
                    dropout=mlp_dropout,
                ))
        self.last_act = get_act(latent_last_act)

    def forward(self, x, t, **kwargs):
        t = timestep_embedding(t, self.time_emb_dim_init)
        cond = self.time_embed(t)
        h = x
        for i in range(len(self.layers)):
            if i in self.skip_layers:
                # injecting input into the hidden layers
                h = torch.cat([h, x], dim=1)
            h = self.layers[i].forward(x=h, cond=cond)
        h = self.last_act(h)
        return Return(pred=h)

class Classifier(nn.Module):
    def __init__(self, block_size=16, last_channel_ratio=1, num_cls=None, z_normalize=True):
        super().__init__()
        self.feature_channel = get_encode_feature_channel(block_size, last_channel_ratio)
        self.z_normalize = z_normalize
        self.classifier = nn.Linear(self.feature_channel, num_cls)

    def forward(self, encode_feature):
        if self.z_normalize:
            encode_feature = feature_z_normalize(encode_feature)
        # loss calculated by F.binary_cross_entropy_with_logits(pred, gt) witch non activation for pred
        class_raw_logit = self.classifier(encode_feature)
        return class_raw_logit
    
class MLPLNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool,
        use_cond: bool,
        activation: str,
        cond_channels: int,
        condition_bias: float = 0,
        dropout: float = 0,
    ):
        super().__init__()
        self.activation = activation
        self.condition_bias = condition_bias
        self.use_cond = use_cond

        self.linear = nn.Linear(in_channels, out_channels)
        self.act = get_act(activation)
        if self.use_cond:
            self.linear_emb = nn.Linear(cond_channels, out_channels)
            self.cond_layers = nn.Sequential(self.act, self.linear_emb)
        if norm:
            self.norm = feature_z_normalize
        else:
            self.norm = nn.Identity()

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation == "relu":
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                elif self.activation == "leakyrelu":
                    init.kaiming_normal_(module.weight,
                                         a=0.2,
                                         nonlinearity='leaky_relu')
                elif self.activation == "silu":
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                else:
                    # leave it as default
                    pass

    def forward(self, x, cond=None):
        x = self.linear(x)
        if self.use_cond:
            # (n, c) or (n, c * 2)
            cond = self.cond_layers(cond)
            cond = (cond, None)

            # scale shift first
            x = x * (self.condition_bias + cond[0])
            if cond[1] is not None:
                x = x + cond[1]
            # then norm
            x = self.norm(x)
        else:
            # no condition
            x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x
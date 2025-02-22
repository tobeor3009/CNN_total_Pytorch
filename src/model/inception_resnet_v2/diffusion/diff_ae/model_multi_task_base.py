from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.nn import init
from .nn import timestep_embedding
from .diffusion_layer import default, Return, EmbedSequential
from .diffusion_layer import ConvBlockND, ResNetBlockND, OutputND
from .diffusion_layer import ResNetBlockNDSkip, ConvBlockNDSkip, MultiDecoderND_V2
from .diffusion_layer import get_maxpool_nd, get_avgpool_nd
from .diffusion_layer import LinearAttention, Attention, AttentionBlock, MaxPool2d, AvgPool2d, MultiInputSequential
from .diffusion_layer import default, prob_mask_like, LearnedSinusoidalPosEmb, SinusoidalPosEmb, GroupNorm32
from .diffusion_layer import feature_z_normalize, z_normalize
from .diffusion_layer import ClassificationHeadSimple
from src.model.inception_resnet_v2.common_module.layers import get_act, get_norm
from einops import rearrange, repeat
from .sub_models import MLPSkipNet, Classifier

def get_encode_feature_channel(block_size, model_depth):
    feature_channel = block_size * (2 ** model_depth)
    return int(round(feature_channel))

def get_time_emb_dim(block_size):
    # emb_dim = block_size * 16
    # I found 512 is best size for all size
    emb_dim = 512
    time_emb_dim_init = emb_dim // 2
    time_emb_dim = emb_dim * 4
    return emb_dim, time_emb_dim_init, time_emb_dim

class BasicUNet(nn.Module):
    def __init__(self, in_channel=3, cond_channel=3, img_size=256, block_size=32,
                 emb_channel=1024, decode_init_channel=None,
                 norm=GroupNorm32, act="silu", num_class_embeds=None, drop_prob=0.05, cond_drop_prob=0.5,
                 self_condition=False, use_checkpoint=[False, False, False, True],
                 attn_info_list=[None, None, None, True], attn_dim_head=32, num_head_list=[1, 1, 1, 1],
                 diffusion_out_channel=1, diffusion_act=None, diffusion_decode_fn_str_list=["conv_transpose", "pixel_shuffle"],
                 seg_out_channel=2, seg_act="softmax", seg_decode_fn_str_list=["conv_transpose", "pixel_shuffle"],
                 class_out_channel=2, class_act="softmax",
                 recon_out_channel=None, recon_act="tanh", recon_decode_fn_str_list=["conv_transpose", "pixel_shuffle"],
                 validity_shape=(1, 8, 8), validity_act=None,
                 get_diffusion=False, get_seg=True, get_class=False, get_recon=False, get_validity=False,
                 include_encoder=False, include_latent_net=False, img_dim=2, encoder_img_dim=None, use_residual_conv=True):
        super().__init__()
        self.include_encoder = include_encoder
        self.include_latent_net = include_latent_net
        self.use_inception_block_attn = True
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
        assert len(attn_info_list) == len(use_checkpoint), "check len(attn_info_list) == len(use_checkpoint)"
        assert len(attn_info_list) == len(num_head_list), "check len(use_checkpoint) == len(num_head_list)"
        assert isinstance(img_dim, int), "img_dim must be int"
        self.model_depth = len(attn_info_list)
        ##################################
        self.use_non_diffusion = get_seg or get_class or get_recon or get_validity
        self.get_diffusion = get_diffusion
        self.get_seg = get_seg
        self.get_class = get_class
        self.get_recon = get_recon
        self.get_validity = get_validity
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
                    self.encoder = BasicEncoder(in_channel=cond_channel, img_size=img_size, block_size=block_size,
                                                emb_channel=emb_channel, norm=norm, act=act,
                                                use_checkpoint=use_checkpoint, attn_info_list=attn_info_list,
                                                attn_dim_head=attn_dim_head, num_head_list=num_head_list,
                                                img_dim=self.encoder_img_dim,
                                                use_residual_conv=use_residual_conv)
                emb_dim_list.append(emb_channel)
                emb_type_list.append("cond")
            if self.include_latent_net:
                if isinstance(include_latent_net, nn.Module):
                    self.latent_net = include_latent_net
                else:
                    self.emb_channel = emb_channel
                    self.latent_net = MLPSkipNet(emb_channel=emb_channel, num_time_layers=2, num_time_emb_channels=64, time_last_act=None,
                                                activation="silu", use_norm=True, num_hid_channels=1024,
                                                num_latent_layers=10, latent_last_act=None, latent_dropout=0., latent_condition_bias=1,
                                                skip_layers=[1, 2, 3, 4, 5, 6, 7, 8, 9])
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
        self.feature_channel = get_encode_feature_channel(block_size, self.model_depth)
        if decode_init_channel is None:
            decode_init_channel = self.feature_channel
        self.decode_init_channel = decode_init_channel

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
        
        if infer_diffusion and self.get_diffusion:
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
        non_diffusion_emb_list = [None]
        if self.include_encoder:
            latent_feature = self.encoder(x)
            non_diffusion_emb_list.append(latent_feature)
        else:
            non_diffusion_emb_list.append(None)
        if class_emb is None:
            non_diffusion_emb_list.append(class_emb)

        encode_feature, skip_connect_list = self.encode_forward(x, *non_diffusion_emb_list)
        if self.get_seg:
            seg_decode_feature = self.decode_forward(self.seg_decoder_list, encode_feature, skip_connect_list, *non_diffusion_emb_list)
            output["seg_pred"] = seg_decode_feature
        if self.get_class:
            class_output = self.class_head(encode_feature)
            output["class_pred"] = class_output
        if self.get_recon:
            recon_decode_feature = self.decode_forward(self.recon_decoder_list, encode_feature, skip_connect_list, *non_diffusion_emb_list)
            output["recon_pred"] = recon_decode_feature
        if self.get_validity:
            validitiy_output = self.validity_head(encode_feature)
            output["validity_pred"] = validitiy_output
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
    
    def encode_forward(self, x, *emb_args):
        # skip connection name list
        # ["stem_layer_1", "stem_layer_4", "stem_layer_7", "mixed_6a", "mixed_7a"]
        skip_connect_list = []
        x = self.init_block(x, *emb_args)
        skip_connect_list.append(x)
        
        for encode_block in self.encode_block_list:
            x = encode_block(x, *emb_args)
            skip_connect_list.append(x)
        x = self.encode_final_block(x, *emb_args)

        return x, skip_connect_list[::-1]
    
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
        
    def decode_forward(self, decoder_list, encode_feature, skip_connect_list, *args):
        decode_block_list, decode_layer_up_list, decode_final_conv = decoder_list
        decode_feature = encode_feature
        for decode_idx, (decode_block, decode_layer_up) in enumerate(zip(decode_block_list, decode_layer_up_list)):
            skip = skip_connect_list[decode_idx]
            decode_feature = decode_block(decode_feature, skip, *args)
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
        get_attn_info = self.get_attn_info
        attn_info_list = self.attn_info_list
        attn_dim_head = self.attn_dim_head
        emb_dim_list = self.emb_dim_list
        emb_type_list = self.emb_type_list
        norm = self.norm
        act = self.act
        use_checkpoint = self.use_checkpoint
        feature_channel = self.feature_channel
        conv_block = self.conv_block

        common_kwarg_dict = self.get_common_kwarg_dict(use_checkpoint=None, is_diffusion=True)
        common_kwarg_dict["kernel_size"] = 3
        common_kwarg_dict["padding"] = 1
        # Stem block

        init_block_0 = conv_block(self.in_channel, block_size, stride=1,
                                attn_info=None,
                                use_checkpoint=self.use_checkpoint[0], image_shape=self.get_image_shape(0), **common_kwarg_dict)
        init_block_1 = conv_block(block_size, block_size, stride=1,
                                attn_info=get_attn_info(emb_type_list, attn_info_list[0], self.attn_dim_head[0]),
                                use_checkpoint=self.use_checkpoint[0], image_shape=self.get_image_shape(0), **common_kwarg_dict)

        encode_block_list = []
        for encode_idx, (attn_info, attn_dim_head, use_checkpoint) in enumerate(zip(self.attn_info_list, self.attn_dim_head, self.use_checkpoint)):
            
            encoder_in_channel = block_size * (2 ** encode_idx)
            encoder_out_channel = encoder_in_channel * 2
            
            encode_block_0 = conv_block(encoder_in_channel, encoder_out_channel, stride=1,
                            attn_info=None,
                            use_checkpoint=use_checkpoint, image_shape=self.get_image_shape(encode_idx), **common_kwarg_dict)
            encode_block_1 = conv_block(encoder_out_channel, encoder_out_channel, stride=2,
                            attn_info=get_attn_info(emb_type_list, attn_info, attn_dim_head),
                            use_checkpoint=use_checkpoint, image_shape=self.get_image_shape(encode_idx), **common_kwarg_dict)
            encode_block = EmbedSequential(encode_block_0, encode_block_1)
            encode_block_list.append(encode_block)

        layer_idx = -1
        encode_final_block = conv_block(feature_channel, feature_channel, 3,
                                            norm=norm, act=act, emb_dim_list=emb_dim_list, emb_type_list=emb_type_list, attn_info=None,
                                            use_checkpoint=self.use_checkpoint[layer_idx], image_shape=self.get_image_shape(self.model_depth), img_dim=self.img_dim)        
        encode_final_attn = self.get_attn_layer(feature_channel, self.num_head_list[layer_idx], self.attn_dim_head[layer_idx],
                                                     True, use_checkpoint=self.use_checkpoint[layer_idx])
        encode_final_block2 = conv_block(feature_channel, feature_channel, 3,
                                                norm=norm, act=act, emb_dim_list=emb_dim_list, emb_type_list=emb_type_list, attn_info=None,
                                                use_checkpoint=self.use_checkpoint[layer_idx], image_shape=self.get_image_shape(self.model_depth), img_dim=self.img_dim)
        
        self.init_block = EmbedSequential(init_block_0, init_block_1)
        self.encode_block_list = nn.ModuleList(encode_block_list)
        self.encode_final_block = EmbedSequential(encode_final_block, encode_final_attn, encode_final_block2)


    def get_decoder(self, decode_out_channel, decode_out_act, decode_fn_str_list, is_diffusion=False):
        decode_init_channel = self.decode_init_channel
        attn_info_list = self.attn_info_list
        num_head_list = self.num_head_list
        use_checkpoint = self.use_checkpoint

        decoder_block_list = []
        decoder_layer_up_list = []
        for decode_idx in range(self.model_depth):
            if decode_idx == 0:
                decode_in_channel = self.feature_channel
            else:
                decode_in_channel = decode_init_channel // (2 ** decode_idx)
            decode_block_out_channel = decode_init_channel // (2 ** (decode_idx + 1))

            decode_idx = self.model_depth - decode_idx - 1
            attn_info = self.get_attn_info(attn_info_list[decode_idx], num_head_list[decode_idx])
            common_kwarg_dict = self.get_common_kwarg_dict(use_checkpoint=use_checkpoint[decode_idx], is_diffusion=is_diffusion)
            common_kwarg_dict["attn_info"] = attn_info
            common_kwarg_dict["image_shape"] = self.get_image_shape(decode_idx + 1)
            skip_channel = self.block_size * (2 ** (decode_idx + 1))
            decoder_block = self.conv_skip_block(decode_in_channel + skip_channel, decode_block_out_channel,
                                                kernel_size=3, stride=1, **common_kwarg_dict)

            decoder_layer_up = MultiDecoderND_V2(decode_block_out_channel, decode_block_out_channel,
                                                kernel_size=2, decode_fn_str_list=decode_fn_str_list,
                                                use_residual_conv=self.use_residual_conv, **common_kwarg_dict)
            decoder_block_list.append(decoder_block)
            decoder_layer_up_list.append(decoder_layer_up)
        decoder_block_list = nn.ModuleList(decoder_block_list)
        decoder_layer_up_list = nn.ModuleList(decoder_layer_up_list)
        decode_final_conv = OutputND(decode_block_out_channel, decode_out_channel,
                                     act=decode_out_act, img_dim=self.img_dim)
        return nn.ModuleList([decoder_block_list, decoder_layer_up_list, decode_final_conv])

    def get_validity_block(self, validity_shape, validity_act):
        validity_init_channel = self.block_size * 8
        common_kwarg_dict = self.get_common_kwarg_dict(use_checkpoint=self.use_checkpoint[self.model_depth - 1],
                                                       is_diffusion=False)
        common_kwarg_dict["kernel_size"] = 3
        common_kwarg_dict["padding"] = 1
        common_kwarg_dict["image_shape"] = self.get_image_shape(self.model_depth)
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

class BasicEncoder(BasicUNet):
    def __init__(self, in_channel, img_size, block_size=32, emb_channel=1024, drop_prob=0.0,
                 norm=GroupNorm32, act="silu",
                 use_checkpoint=False, attn_info_list=[None, False, False, True],
                 attn_dim_head=32, num_head_list=[1, 1, 1, 1],
                 img_dim=2, use_residual_conv=False):
        super(BasicUNet, self).__init__()

        self.use_inception_block_attn = True
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
        self.model_depth = len(attn_info_list)
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
        self.feature_channel = get_encode_feature_channel(block_size, self.model_depth)
        ##################################
        if use_residual_conv:
            conv_block = ResNetBlockND
        else:
            conv_block = ConvBlockND
        self.use_residual_conv = use_residual_conv
        self.conv_block = conv_block
        self.image_shape = self.get_image_init_shape()
        ##################################
        self.set_encoder()
        if self.image_shape is not None:
            image_shape = self.get_image_shape(5)
            norm_shape = (self.feature_channel, *image_shape)
        else:
            norm_shape = self.feature_channel

        self.pool_layer = nn.Sequential(
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
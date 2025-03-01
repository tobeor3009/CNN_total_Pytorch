from timm.models.layers import trunc_normal_
import torch
import numpy as np
from functools import partial
from torch import nn
from einops.layers.torch import Rearrange
from .nn import timestep_embedding
from .diffusion_layer import OutputND, ConvBlockND, Return, GroupNorm32
from src.model.swin_transformer.model_2d.swin_layers import Mlp
from .diffusion_layer_swin_2d import AttentionPool1d, RMSNorm, AttentionBlock
from .diffusion_layer_swin_2d import BasicLayerV1, BasicLayerV2, MeanBN2BC
from .diffusion_layer_swin_2d import default, prob_mask_like, get_act, get_norm
from .diffusion_layer_swin_2d import PatchEmbed, PatchMerging, PatchExpanding, BasicDecodeLayer
from .model import MLPSkipNet
from einops import rearrange, repeat

def get_time_emb_dim(emb_dim):
    # time_emb_dim = max(512, emb_dim * 8)
    # I found 512 is best size. 256 is not enough, 1024 is too big so cosumes too much memory
    time_emb_dim = 512
    time_emb_dim_init = time_emb_dim // 4
    return time_emb_dim_init, time_emb_dim

class SwinMultitask(nn.Module):
    def __init__(self, img_size=512, patch_size=4,
                 in_channel=3, cond_channel=3, self_condition=False,
                 norm_layer="rms", act_layer="silu", emb_channel=1024,
                 diffusion_out_channel=1, diffusion_out_act=None, diffusion_decode_fn_str="pixel_shuffle",
                 seg_out_channel=2, seg_out_act="softmax", seg_decode_fn_str="pixel_shuffle",
                 num_classes=1000, class_act="softmax", recon_act="sigmoid", recon_decode_fn_str="pixel_shuffle",
                 validity_shape=(1, 8, 8), validity_act=None,
                num_class_embeds=None, cond_drop_prob=0.5,
                embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                window_sizes=[8, 4, 4, 2], mlp_ratio=4., qkv_bias=True, ape=True, patch_norm=True,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                use_checkpoint=False, pretrained_window_sizes=0,
                get_diffusion=False, get_seg=True, get_class=False, get_recon=False, get_validity=False,
                include_encoder=False, include_latent_net=False):
        super().__init__()
        norm_layer_list = ["rms", "group"]
        assert norm_layer in norm_layer_list, f"you can choose norm layer: {norm_layer_list}"
        self.norm_layer = norm_layer
        self.model_act_layer = act_layer
        patch_size = int(patch_size)
        self.img_dim = 2
        # for compability with diffusion_sample
        self.img_size = img_size
        self.cond_channel = cond_channel
        self.out_channel = diffusion_out_channel
        self.self_condition = self_condition
        if self.self_condition:
            in_channel *= 2
        ##################################
        self.patch_size = patch_size
        self.patch_norm = patch_norm
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_sizes = window_sizes
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.num_class_embeds = num_class_embeds
        self.cond_drop_prob = cond_drop_prob
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        if isinstance(pretrained_window_sizes, int):
            pretrained_window_sizes = [pretrained_window_sizes for _ in num_heads]
        self.pretrained_window_sizes = pretrained_window_sizes
        if isinstance(use_checkpoint, bool):
            use_checkpoint = [use_checkpoint for _ in num_heads]
        self.use_checkpoint = use_checkpoint
        self.use_non_diffusion = get_seg or get_class or get_recon or get_validity
        self.get_diffusion = get_diffusion
        self.get_seg = get_seg
        self.get_class = get_class
        self.get_recon = get_recon
        self.get_validity = get_validity
        ##################################
        self.num_layers = len(depths)
        self.ape = ape
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.feature_dim = int(self.embed_dim * 2 ** self.num_layers)
        # for compability with diffusion_sample
        self.img_size = img_size
        self.in_channel = in_channel
        self.cond_channel = cond_channel
        self.out_channel = diffusion_out_channel
        self.emb_channel = emb_channel
        self.self_condition = self_condition
        if self.self_condition:
            in_channel *= 2
        # stochastic depth
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                    in_chans=in_channel, embed_dim=embed_dim,
                                    norm_layer=self.norm_layer if self.patch_norm else None)
        
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.feature_hw = np.array(self.patches_resolution) // (2 ** self.num_layers)

        if self.ape:
            pos_embed_shape = torch.zeros(1, num_patches, embed_dim)
            self.absolute_pos_embed = nn.Parameter(pos_embed_shape)
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        emb_dim_list = []
        emb_type_list = []
        time_emb_dim_init, time_emb_dim = get_time_emb_dim(embed_dim)
        self.include_encoder = include_encoder
        self.include_latent_net = include_latent_net
        if get_diffusion:
            self.time_emb_dim_init = time_emb_dim_init
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim_init, time_emb_dim),
                get_act(act_layer),
                nn.Linear(time_emb_dim, time_emb_dim)
            )
            emb_dim_list.append(time_emb_dim)
            emb_type_list.append("seq")

            if self.include_encoder:
                if isinstance(include_encoder, nn.Module):
                    self.encoder = include_encoder
                else:
                    self.encoder = SwinEncoder(img_size=img_size, patch_size=patch_size,
                                                in_channel=cond_channel, emb_channel=emb_channel, self_condition=False,
                                                norm_layer=norm_layer, act_layer=act_layer,
                                                embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                                window_sizes=window_sizes, mlp_ratio=mlp_ratio,
                                                qkv_bias=qkv_bias, ape=ape, patch_norm=patch_norm,
                                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                                                use_checkpoint=use_checkpoint, pretrained_window_sizes=pretrained_window_sizes)
                emb_dim_list.append(emb_channel)
                emb_type_list.append("cond")
            if self.include_latent_net:
                if isinstance(include_latent_net, nn.Module):
                    self.latent_net = include_latent_net
                else:
                    self.emb_channel = emb_channel
                    self.latent_net = MLPSkipNet(emb_channel=emb_channel, block_size=16, num_time_layers=2, time_last_act=None,
                                                num_latent_layers=10, latent_last_act=None, latent_dropout=0., latent_condition_bias=1,
                                                act="silu", use_norm=True, skip_layers=[1, 2, 3, 4, 5, 6, 7, 8, 9])

        if num_class_embeds is not None:
            class_emb_dim = time_emb_dim
            self.class_emb_layer = nn.Embedding(num_class_embeds, class_emb_dim)
            if self.cond_drop_prob > 0:
                self.null_class_emb = nn.Parameter(torch.randn(class_emb_dim))
            else:
                self.null_class_emb = None
            
            self.class_mlp = nn.Sequential(
                nn.Linear(class_emb_dim, class_emb_dim),
                get_act(self.model_act_layer),
                nn.Linear(class_emb_dim, class_emb_dim)
            )
            emb_dim_list.append(class_emb_dim)
            emb_type_list.append("seq")
        else:
            self.class_emb_layer = None
            self.null_class_emb = None
            self.class_mlp = None
        self.emb_dim_list = emb_dim_list
        self.emb_type_list = emb_type_list
        ###################### Define Encoder ######################
        self.init_layer = self.get_init_layer()
        self.encode_layers = self.get_encode_layers()
        self.mid_layer_1, self.mid_attn, self.mid_layer_2 = self.get_mid_layer()
        ###################### Define Parts ########################
        if get_diffusion:
            self.diff_decode_layers = self.get_decode_layers(decode_fn_str=diffusion_decode_fn_str, is_diffusion=True)
            diffusion_layer_list = self.get_decode_final_layers(diffusion_out_channel, diffusion_out_act, is_diffusion=True)
            self.diffusion_layer_list = nn.ModuleList(diffusion_layer_list)
        if get_seg:
            self.seg_decode_layers = self.get_decode_layers(decode_fn_str=seg_decode_fn_str)
            seg_layer_list = self.get_decode_final_layers(seg_out_channel, seg_out_act)
            self.seg_layer_list = nn.ModuleList(seg_layer_list)
        if get_class:
            self.class_head = self.get_class_head(num_classes, class_act)
        if get_recon:
            self.recon_decode_layers = self.get_decode_layers(decode_fn_str=recon_decode_fn_str)
            recon_layer_list = self.get_decode_final_layers(in_channel, recon_act)
            self.recon_layer_list = nn.ModuleList(recon_layer_list)
        if get_validity:
            self.validity_dim = int(embed_dim * (2 ** self.num_layers))
            self.validity_head = self.get_validity_head(validity_shape, validity_act)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}
    
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
        
        time_emb = timestep_embedding(t, self.time_emb_dim_init)
        time_emb = self.time_mlp(time_emb)
        emb_list.append(time_emb)
        if self.encoder is not None:
            if cond is None:
                latent_emb = self.encoder(x_start)
            else:
                latent_emb = cond
            emb_list.append(latent_emb)

        if class_emb is not None:
            emb_list.append(class_emb)

        diff_feature, diff_skip_connect_list = self.process_encode_layers(x, emb_list)
        diff_feature = self.process_mid_layers(diff_feature, emb_list)
        diff_output = self.process_decode_layers(diff_feature, self.diff_decode_layers,
                                                 self.diffusion_layer_list, diff_skip_connect_list, emb_list)
        output["pred"] = diff_output
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

        encoded_feature, skip_connect_list = self.process_encode_layers(x, non_diffusion_emb_list)
        encoded_feature = self.process_mid_layers(encoded_feature, non_diffusion_emb_list)
        if self.get_seg:
            seg_output = self.process_decode_layers(encoded_feature, self.seg_decode_layers,
                                                    self.seg_layer_list, skip_connect_list, non_diffusion_emb_list)
            output["seg_pred"] = seg_output
        if self.get_class:
            class_output = self.class_head(encoded_feature)
            output["class_pred"] = class_output
        if self.get_recon:
            recon_output = self.process_decode_layers(encoded_feature, self.recon_decode_layers,
                                                    self.recon_layer_list, skip_connect_list, non_diffusion_emb_list)
            output["recon_pred"] = recon_output
        if self.get_validity:
            validitiy_output = self.validity_head(encoded_feature)
            output["validity_pred"] = validitiy_output
        return output
    
    def process_class_emb(self, x, class_labels, cond_drop_prob):
        if self.num_class_embeds is not None:
            cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
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
    
    def process_encode_layers(self, x, emb_list=[]):
        skip_connect_list = []
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.init_layer(x, *emb_list)
        skip_connect_list.append(x)
        
        for encode_idx, encode_layer in enumerate(self.encode_layers):
            x = encode_layer(x, *emb_list)
            if encode_idx < self.num_layers - 1:
                skip_connect_list.append(x)
            
        return x, skip_connect_list[::-1]

    def process_mid_layers(self, x, emb_list=[]):
        x = self.mid_layer_1(x, *emb_list)
        x = self.mid_attn(x)
        x = self.mid_layer_2(x, *emb_list)
        return x

    def process_decode_layers(self, x, decode_layers,
                              decode_layer_list, skip_connect_list, emb_list=[]):
        decode_final_layer_1, decode_final_layer_2, decode_final_expanding, decode_out_conv = decode_layer_list
        for decode_idx, decode_layer in enumerate(decode_layers, start=0):
            skip_x = skip_connect_list[decode_idx]
            x = decode_layer(x, skip_x, *emb_list)

        x = decode_final_layer_1(x, skip_connect_list[-1], *emb_list)
        x = decode_final_layer_2(x, *emb_list)
        x = decode_final_expanding(x)
        x = decode_out_conv(x)
        return x
    
    
    def get_layer_config_dict(self, dim, feature_resolution, i_layer, is_diffusion):
        depths = self.depths
        if is_diffusion:
            emb_dim_list = self.emb_dim_list
            emb_type_list = self.emb_type_list
        else:
            if self.num_class_embeds is None:
                emb_dim_list = []
                emb_type_list = []
            else:
                emb_dim_list = self.emb_dim_list[-1:]
                emb_type_list = self.emb_type_list[-1:]
                
        common_kwarg_dict = {"dim":dim,
                            "input_resolution":feature_resolution,
                            "depth":depths[i_layer],
                            "num_heads":self.num_heads[i_layer],
                            "window_size":self.window_sizes[i_layer],
                            "mlp_ratio":self.mlp_ratio, "qkv_bias":self.qkv_bias, 
                            "drop":self.drop_rate, "attn_drop":self.attn_drop_rate,
                            "drop_path":self.dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                            "norm_layer":self.norm_layer,
                            "act_layer":get_act(self.model_act_layer),
                            "pretrained_window_size":self.pretrained_window_sizes[i_layer],
                            "use_checkpoint":self.use_checkpoint[i_layer],
                            "emb_dim_list": emb_dim_list,
                            "emb_type_list": emb_type_list,
                            "img_dim": self.img_dim
                            }
        return common_kwarg_dict
    
    def get_init_layer(self):
        common_kwarg_dict = self.get_layer_config_dict(self.embed_dim, self.patches_resolution, 0, is_diffusion=True)
        init_layer = BasicLayerV2(**common_kwarg_dict)
        return init_layer
    
    def get_encode_layers(self):
        encode_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_dim = int(self.embed_dim * 2 ** i_layer)
            feature_resolution = np.array(self.patches_resolution) // (2 ** i_layer)
            common_kwarg_dict = self.get_layer_config_dict(layer_dim, feature_resolution, i_layer, is_diffusion=True)
            common_kwarg_dict["downsample"] = PatchMerging
            encode_layer = BasicLayerV2(**common_kwarg_dict)
            encode_layers.append(encode_layer)
        return encode_layers
    
    def get_mid_layer(self):
        i_layer = self.num_layers - 1
        feature_dim = self.feature_dim
        common_kwarg_dict = self.get_layer_config_dict(feature_dim, self.feature_hw, i_layer, is_diffusion=True)
        mid_layer_1 = BasicLayerV2(**common_kwarg_dict)
        mid_attn_norm_layer = "group" if self.norm_layer == "group" else "rms"
        # TBD: self.attn_drop_rate 추가할지 고민
        mid_attn = AttentionBlock(channels=feature_dim, num_heads=common_kwarg_dict["num_heads"],
                                  use_checkpoint=common_kwarg_dict["use_checkpoint"], norm_layer=mid_attn_norm_layer)
        mid_layer_2 = BasicLayerV2(**common_kwarg_dict)
        return mid_layer_1, mid_attn, mid_layer_2
    
    def get_decode_layers(self, decode_fn_str="pixel_shuffle", is_diffusion=False):
        decode_layers = nn.ModuleList()
        for d_i_layer in range(self.num_layers, 0, -1):
            i_layer = d_i_layer - 1
            layer_dim = int(self.embed_dim * 2 ** d_i_layer)
            feature_resolution = np.array(self.patches_resolution) // (2 ** d_i_layer)
            common_kwarg_dict = self.get_layer_config_dict(layer_dim, feature_resolution, i_layer, is_diffusion)
            common_kwarg_dict["skip_dim"] = layer_dim // 2
            common_kwarg_dict["upsample"] = PatchExpanding
            common_kwarg_dict["decode_fn_str"] = decode_fn_str
            decode_layer = BasicDecodeLayer(**common_kwarg_dict)
            decode_layers.append(decode_layer)
        return decode_layers
    
    def get_decode_final_layers(self, decode_out_channel, decode_out_act, is_diffusion=False):
        
        i_layer = 0
        common_kwarg_dict = self.get_layer_config_dict(self.embed_dim, self.patches_resolution, i_layer, is_diffusion)
        decode_final_layer_1 = BasicDecodeLayer(skip_dim=self.embed_dim, upsample=None, **common_kwarg_dict)
        decode_final_layer_2 = BasicLayerV1(**common_kwarg_dict)
        decode_final_expanding = PatchExpanding(input_resolution=self.patches_resolution,
                                                    dim=self.embed_dim,
                                                    return_vector=False,
                                                    dim_scale=self.patch_size,
                                                    norm_layer=self.norm_layer,
                                                    decode_fn_str="pixel_shuffle",
                                                    img_dim=self.img_dim
                                                    )
        decode_out_conv = OutputND(self.embed_dim // 2, decode_out_channel, act=decode_out_act, img_dim=self.img_dim)
        return decode_final_layer_1, decode_final_layer_2, decode_final_expanding, decode_out_conv
    
    def get_class_head(self, num_classes, class_act):
        feature_dim = self.feature_dim
        class_head = nn.Sequential(
                get_act(self.model_act_layer),
                AttentionPool1d(sequence_length=np.prod(self.feature_hw), embed_dim=feature_dim,
                                num_heads=8, output_dim=feature_dim * 2, channel_first=False),
                nn.Linear(feature_dim * 2, feature_dim),
                nn.Dropout(p=0.1),
                get_act(self.model_act_layer),
                nn.Linear(feature_dim, num_classes),
                get_act(class_act)
        )
        return class_head
    
    def get_validity_head(self, validity_shape, validity_act):
        i_layer = 0
        h, w = self.feature_hw
        common_kwarg_dict = self.get_layer_config_dict(self.feature_dim, self.feature_hw, i_layer, is_diffusion=False)
        common_kwarg_dict["emb_dim_list"] = []
        common_kwarg_dict["emb_type_list"] = []
        validity_layer_1 = BasicLayerV2(**common_kwarg_dict)
        validity_layer_2 = BasicLayerV2(**common_kwarg_dict)
        validity_mlp = Mlp(self.feature_dim, hidden_features=self.feature_dim // 2,
                           out_features=self.feature_dim, act_layer=get_act(self.model_act_layer))
        if self.img_dim == 1:
            pool_layer = nn.AdaptiveAvgPool1d
        elif self.img_dim == 2:
            pool_layer = nn.AdaptiveAvgPool2d
        elif self.img_dim == 3:
            pool_layer = nn.AdaptiveAvgPool3d
        validity_conv = ConvBlockND(in_channels=self.feature_dim, out_channels=validity_shape[0],
                                    kernel_size=3, stride=1, padding=1, norm=GroupNorm32, act=self.model_act_layer,
                                    dropout_proba=self.drop_rate)
        validity_avg_pool = pool_layer(validity_shape[1:])
        validity_head = nn.Sequential(
            validity_layer_1,
            validity_layer_2,
            validity_mlp,
            Rearrange('b (h w) c -> b c h w', h=h, w=w),
            validity_conv,
            validity_avg_pool,
            get_act(validity_act)
        )
        return validity_head

class SwinEncoder(SwinMultitask):
    def __init__(self, img_size=512, patch_size=4,
                in_channel=3, emb_channel=1024, self_condition=False,
                norm_layer="rms", act_layer="silu",
                embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                window_sizes=[8, 4, 4, 2], mlp_ratio=4., qkv_bias=True, ape=True, patch_norm=True,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                use_checkpoint=False, pretrained_window_sizes=0):
        super(SwinMultitask, self).__init__()
        self.norm_layer = norm_layer
        self.model_act_layer = act_layer
        patch_size = int(patch_size)
        self.img_dim = 2
        # for compability with diffusion_sample
        self.img_size = img_size
        self.self_condition = self_condition
        if self.self_condition:
            in_channel *= 2
        ##################################
        self.patch_size = patch_size
        self.patch_norm = patch_norm
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_sizes = window_sizes
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        ##################################
        self.num_layers = len(depths)
        self.ape = ape
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.feature_dim = int(self.embed_dim * 2 ** self.num_layers)
        if isinstance(pretrained_window_sizes, int):
            pretrained_window_sizes = [pretrained_window_sizes for _ in num_heads]
        self.pretrained_window_sizes = pretrained_window_sizes
        if isinstance(use_checkpoint, bool):
            use_checkpoint = [use_checkpoint for _ in num_heads]
        self.use_checkpoint = use_checkpoint
        # stochastic depth
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        self.emb_dim_list = []
        self.emb_type_list = []
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                    in_chans=in_channel, embed_dim=embed_dim,
                                    norm_layer=self.norm_layer if self.patch_norm else None)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.feature_hw = np.array(self.patches_resolution) // (2 ** self.num_layers)
        if self.ape:
            pos_embed_shape = torch.zeros(1, num_patches, embed_dim)
            self.absolute_pos_embed = nn.Parameter(pos_embed_shape)
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.init_layer = self.get_init_layer()
        self.encode_layers = self.get_encode_layers()
        self.mid_layer_1, self.mid_attn, self.mid_layer_2 = self.get_mid_layer()
        self.pool_layer = nn.Sequential(
            get_norm(norm_layer, self.feature_dim),
            get_act(act_layer),
            MeanBN2BC(target_dim=1),
            nn.Flatten(start_dim=1),
            nn.Linear(self.feature_dim, emb_channel)
        )

    def forward(self, x):
        x, _ = self.process_encode_layers(x)
        x = self.process_mid_layers(x)
        x = self.pool_layer(x)
        return x
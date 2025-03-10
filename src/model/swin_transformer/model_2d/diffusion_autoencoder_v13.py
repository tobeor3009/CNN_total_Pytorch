from timm.models.layers import trunc_normal_
import torch
import numpy as np
from functools import partial
from torch import nn
from torch.utils.checkpoint import checkpoint
from .swin_layers import get_act, Output2D, InstanceNormChannelLast
from .swin_layers_diffusion_2d import LinearAttention2D, LinearAttention, Attention2D, Attention, RMSNorm
from .swin_layers_diffusion_2d import BasicLayerV1, BasicLayerV2, SkipConv1D, SkipLinear, AttentionPool1d
from .swin_layers_diffusion_2d import default, prob_mask_like, LearnedSinusoidalPosEmb
from .swin_layers_diffusion_2d import PatchEmbed, PatchMerging, PatchMergingConv, PatchExpandingLinear, ConvBlock2D
from .swin_layers_diffusion_2d import get_norm_layer_partial, get_norm_layer_partial_conv

from einops import rearrange, repeat

class SwinDiffusionUnet(nn.Module):
    def __init__(self, img_size=512, patch_size=4, in_chans=1,
                 norm_layer="instance", act_layer="relu6",
                 cond_chans=3, out_chans=1, out_act=None,
                 emb_chans=1024, num_class_embeds=None, cond_drop_prob=0.5,
                 embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_sizes=[8, 4, 4, 2], mlp_ratio=4., qkv_bias=True, ape=True, patch_norm=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0, latent_drop_rate=0.1,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0],
                 self_condition=False, use_residual=False):
        super().__init__()
        norm_layer_list = ["rms", "instance"]
        assert norm_layer in norm_layer_list, f"you can choose norm layer: {norm_layer_list}"
        if norm_layer == "rms":
            self.model_norm_layer = RMSNorm
        elif norm_layer == "instance":
            self.model_norm_layer = InstanceNormChannelLast
        
        self.norm_layer = norm_layer
        self.model_act_layer = act_layer

        patch_size = int(patch_size)
        # for compability with Medsegdiff
        self.image_size = img_size
        self.input_img_channels = cond_chans
        self.mask_channels = in_chans
        
        ##################################
        self.cond_drop_prob = cond_drop_prob
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.self_condition = self_condition

        if isinstance(use_checkpoint, bool):
            use_checkpoint = [use_checkpoint for _ in num_heads]
        self.use_checkpoint = use_checkpoint
        if self.self_condition:
            in_chans = in_chans * 2
        
        time_emb_dim = embed_dim * 4
        sinu_pos_emb = LearnedSinusoidalPosEmb(time_emb_dim)
        fourier_dim = time_emb_dim + 1
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_emb_dim),
            get_act(norm_layer),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.latent_encoder = SwinDiffusionEncoder(img_size=img_size, patch_size=patch_size, in_chans=cond_chans, emb_chans=emb_chans,
                                                   norm_layer=norm_layer, act_layer=act_layer,
                                                   embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                                   window_sizes=window_sizes, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, ape=ape,
                                                   drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                                                   use_checkpoint=use_checkpoint, pretrained_window_sizes=pretrained_window_sizes,
                                                   patch_norm=patch_norm, use_residual=use_residual
                                                )


        # class embedding

        self.num_class_embeds = num_class_embeds
        if num_class_embeds is not None:
            class_emb_dim = embed_dim * 4
            self.class_emb_layer = nn.Embedding(num_class_embeds, class_emb_dim)
            if self.cond_drop_prob > 0:
                self.null_class_emb = nn.Parameter(torch.randn(class_emb_dim))
            else:
                self.null_class_emb = None
            
            self.class_mlp = nn.Sequential(
                nn.Linear(class_emb_dim, class_emb_dim),
                get_act(norm_layer),
                nn.Linear(class_emb_dim, class_emb_dim)
            )
            emb_dim_list = [time_emb_dim, emb_chans, time_emb_dim]
        else:
            self.class_emb_layer = None
            self.null_class_emb = None
            self.class_mlp = None
            emb_dim_list = [time_emb_dim, emb_chans]

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                    in_chans=in_chans, embed_dim=embed_dim,
                                    norm_layer=self.model_norm_layer if self.patch_norm else None)
        
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            pos_embed_shape = torch.zeros(1, num_patches, embed_dim)
            self.absolute_pos_embed = nn.Parameter(pos_embed_shape)
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.latent_drop = nn.Dropout(p=latent_drop_rate)

        i_layer = 0
        common_kwarg_dict = {"dim":embed_dim,
                            "input_resolution":patches_resolution,
                            "depth":depths[i_layer],
                            "num_heads":num_heads[i_layer],
                            "window_size":window_sizes[i_layer],
                            "mlp_ratio":self.mlp_ratio, "qkv_bias":qkv_bias, 
                            "drop":drop_rate, "attn_drop":attn_drop_rate,
                            "drop_path":dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                            "norm_layer":self.model_norm_layer,
                            "act_layer":get_act(self.model_act_layer),
                            "pretrained_window_size":pretrained_window_sizes[i_layer],
                            "emb_dim_list":emb_dim_list[:2],
                            "use_checkpoint":use_checkpoint[0], "use_residual":use_residual
                            }
        self.init_layer = BasicLayerV2(**common_kwarg_dict)

        # build layers
        self.encode_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_dim = int(embed_dim * 2 ** i_layer)
            feature_resolution = np.array(patches_resolution) // (2 ** i_layer)
            if i_layer < self.num_layers - 2:
                layer_emb_dim_list = emb_dim_list[:2]
            else:
                layer_emb_dim_list = emb_dim_list
            common_kwarg_dict = {
                                "dim":layer_dim,
                                "input_resolution":feature_resolution,
                                "depth":depths[i_layer],
                               "num_heads":num_heads[i_layer],
                               "window_size":window_sizes[i_layer],
                               "mlp_ratio":self.mlp_ratio, "qkv_bias":qkv_bias, 
                               "drop":drop_rate, "attn_drop":attn_drop_rate,
                               "drop_path":dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               "norm_layer":self.model_norm_layer,
                               "act_layer":get_act(self.model_act_layer),
                                "downsample":PatchMerging,
                               "pretrained_window_size":pretrained_window_sizes[i_layer],
                               "emb_dim_list":layer_emb_dim_list,
                               "use_checkpoint":use_checkpoint[i_layer], "use_residual":use_residual
                               }
            encode_layer = BasicLayerV2(**common_kwarg_dict)
            self.encode_layers.append(encode_layer)

        depth_level = self.num_layers
        layer_dim = int(embed_dim * 2 ** depth_level)
        feature_hw = np.array(patches_resolution) // (2 ** depth_level)
        
        common_kwarg_dict = {"dim":layer_dim,
                            "depth":depths[i_layer],
                           "input_resolution":feature_hw,
                            "num_heads":num_heads[i_layer],
                            "window_size":window_sizes[i_layer],
                            "mlp_ratio":self.mlp_ratio, "qkv_bias":qkv_bias, 
                            "drop":drop_rate, "attn_drop":attn_drop_rate,
                            "drop_path":dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                            "norm_layer":self.model_norm_layer,
                            "act_layer":get_act(self.model_act_layer),
                            "pretrained_window_size":pretrained_window_sizes[i_layer],
                            "emb_dim_list":emb_dim_list,
                            "use_checkpoint":use_checkpoint[i_layer], "use_residual":use_residual
                            }

        self.mid_layer_1 = BasicLayerV2(**common_kwarg_dict)
        mid_attn_norm_layer = "instance" if self.norm_layer == "instance" else "rms"
        self.mid_attn = Attention(dim=layer_dim, num_heads=num_heads[i_layer],
                                  use_checkpoint=use_checkpoint[i_layer], norm_layer=mid_attn_norm_layer)
        self.mid_layer_2 = BasicLayerV2(**common_kwarg_dict)
        
        self.skip_layers = nn.ModuleList()
        self.decode_layers = nn.ModuleList()
        for d_i_layer in range(self.num_layers, 0, -1):
            i_layer = d_i_layer - 1
            layer_dim = int(embed_dim * 2 ** d_i_layer)
            feature_resolution = np.array(patches_resolution) // (2 ** d_i_layer)

            skip_layer = SkipLinear(layer_dim * 2, layer_dim,
                                    norm=self.model_norm_layer)
            
            if d_i_layer > self.num_layers - 2:
                layer_emb_dim_list = emb_dim_list
            else:
                layer_emb_dim_list = emb_dim_list[:2]
            common_kwarg_dict = {"dim":layer_dim,
                               "input_resolution":feature_resolution,
                                "depth":depths[i_layer],
                               "num_heads":num_heads[i_layer],
                               "window_size":window_sizes[i_layer],
                               "mlp_ratio":self.mlp_ratio, "qkv_bias":qkv_bias, 
                               "drop":drop_rate, "attn_drop":attn_drop_rate,
                               "drop_path":dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               "norm_layer":self.model_norm_layer,
                               "act_layer":get_act(self.model_act_layer),
                               "upsample":PatchExpandingLinear,
                               "pretrained_window_size":pretrained_window_sizes[i_layer],
                               "emb_dim_list":layer_emb_dim_list,
                               "use_checkpoint":use_checkpoint[i_layer], "use_residual":use_residual
                               }
            
            decode_layer = BasicLayerV1(**common_kwarg_dict)
            self.skip_layers.append(skip_layer)
            self.decode_layers.append(decode_layer)

        self.seg_final_layer = BasicLayerV1(dim=embed_dim,
                                            input_resolution=patches_resolution,
                                            depth=depths[i_layer],
                                            num_heads=num_heads[i_layer],
                                            window_size=window_sizes[i_layer],
                                            mlp_ratio=self.mlp_ratio,
                                            qkv_bias=qkv_bias,
                                            drop=drop_rate, attn_drop=attn_drop_rate,
                                            drop_path=dpr[sum(depths[:i_layer]):sum(
                                                depths[:i_layer + 1])],
                                            norm_layer=self.model_norm_layer,
                                            act_layer=get_act(self.model_act_layer),
                                            upsample=None,
                                            use_checkpoint=use_checkpoint[i_layer],
                                            pretrained_window_size=pretrained_window_sizes[i_layer],
                                            emb_dim_list=emb_dim_list, use_residual=use_residual)
        self.seg_final_expanding = PatchExpandingLinear(input_resolution=patches_resolution,
                                                    dim=embed_dim,
                                                    return_vector=False,
                                                    dim_scale=patch_size,
                                                    norm_layer=self.model_norm_layer
                                                    )
        self.out_conv = Output2D(embed_dim // 2, out_chans, act=out_act)
        self.apply(self._init_weights)

        for bly in (self.encode_layers):
            bly._init_respostnorm()
        self.mid_layer_1._init_respostnorm()
        self.mid_layer_2._init_respostnorm()
        for bly in (self.decode_layers):
            bly._init_respostnorm()
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

    def forward(self, x, time, cond=None, x_self_cond=None, class_labels=None,
                cond_drop_prob=None):
        
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        batch, device = x.size(0), x.device
        
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        time_emb = self.time_mlp(time)
        latent = self.latent_encoder(cond)
        latent = self.latent_drop(latent)

        if self.num_class_embeds is not None:
            class_emb = self.class_emb_layer(class_labels).to(dtype=x.dtype)
            if cond_drop_prob > 0:
                keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device=device)
                null_classes_emb = repeat(self.null_class_emb, 'd -> b d', b=batch)

                class_emb = torch.where(
                    rearrange(keep_mask, 'b -> b 1'),
                    class_emb, null_classes_emb
                )

            class_emb = self.class_mlp(class_emb)
            emb_list = [time_emb, latent, class_emb]
        else:
            emb_list = [time_emb, latent]

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.init_layer(x, time_emb, latent)

        skip_connect_list = []
        for idx, encode_layer in enumerate(self.encode_layers):
            
            if idx >= 2:
                layer_emb_list = emb_list[:2]
            else:
                layer_emb_list = emb_list
            x = encode_layer(x, *layer_emb_list)
            skip_connect_list.append(x)

        x = self.mid_layer_1(x, *emb_list)
        x = self.mid_attn(x)
        x = self.mid_layer_2(x, *emb_list)

        for idx, (skip_layer, decode_layer) in enumerate(zip(self.skip_layers, self.decode_layers)):
            if idx < 2:
                layer_emb_list = emb_list[:2]
            else:
                layer_emb_list = emb_list
            skip_x = skip_connect_list.pop()
            x = skip_layer(x, skip_x)
            x = decode_layer(x, *layer_emb_list)

        x = self.seg_final_layer(x, *emb_list)
        x = self.seg_final_expanding(x)
        x = self.out_conv(x)
        return x

    def print_tensor_info(self, tensor):
        print(tensor.min(), tensor.max(), torch.isnan(tensor).any())


class SwinDiffusionEncoder(nn.Module):
    def __init__(self, img_size=512, patch_size=4, in_chans=1, emb_chans=1024,
                 norm_layer="instance", act_layer="relu6",
                 embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_sizes=[8, 4, 4, 2], mlp_ratio=4., qkv_bias=True, ape=True, patch_norm=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0],
                 use_residual=False
                ):
        super().__init__()
        if norm_layer == "layer":
            self.model_norm_layer = nn.LayerNorm
        elif norm_layer == "instance":
            self.model_norm_layer = InstanceNormChannelLast
        self.norm_layer = norm_layer
        self.model_act_layer = act_layer
        patch_size = int(patch_size)
        ##################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        if isinstance(use_checkpoint, bool):
            use_checkpoint = [use_checkpoint for _ in num_heads]
        self.use_checkpoint = use_checkpoint
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                    in_chans=in_chans, embed_dim=embed_dim,
                                    norm_layer=self.model_norm_layer)
        
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            pos_embed_shape = torch.zeros(1, num_patches, embed_dim)
            self.absolute_pos_embed = nn.Parameter(pos_embed_shape)
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        i_layer = 0
        common_kwarg_dict = {"dim":embed_dim,
                            "input_resolution":patches_resolution,
                            "depth":depths[i_layer],
                            "num_heads":num_heads[i_layer],
                            "window_size":window_sizes[i_layer],
                            "mlp_ratio":self.mlp_ratio, "qkv_bias":qkv_bias, 
                            "drop":drop_rate, "attn_drop":attn_drop_rate,
                            "drop_path":dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                            "norm_layer":self.model_norm_layer,
                            "act_layer":get_act(self.model_act_layer),
                            "pretrained_window_size":pretrained_window_sizes[i_layer],
                            "emb_dim_list":[],
                            "use_checkpoint":use_checkpoint[i_layer], "use_residual":use_residual
                            }
        self.init_layer = BasicLayerV2(**common_kwarg_dict)

        # build layers
        self.encode_layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer_dim = int(embed_dim * 2 ** i_layer)
            feature_resolution = np.array(patches_resolution) // (2 ** i_layer)
            common_kwarg_dict = {"dim":layer_dim,
                                "input_resolution":feature_resolution,
                                "depth":depths[i_layer],
                               "num_heads":num_heads[i_layer],
                               "window_size":window_sizes[i_layer],
                               "mlp_ratio":self.mlp_ratio, "qkv_bias":qkv_bias, 
                               "drop":drop_rate, "attn_drop":attn_drop_rate,
                               "drop_path":dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               "norm_layer":self.model_norm_layer,
                               "act_layer":get_act(self.model_act_layer),
                               "downsample":PatchMerging,
                               "pretrained_window_size":pretrained_window_sizes[i_layer],
                               "emb_dim_list": [],
                               "use_checkpoint":use_checkpoint[i_layer], "use_residual":use_residual
                               }
            encode_layer = BasicLayerV2(**common_kwarg_dict)
            self.encode_layers.append(encode_layer)

        depth_level = self.num_layers
        layer_dim = int(embed_dim * 2 ** depth_level)
        feature_hw = np.array(patches_resolution) // (2 ** depth_level)
        
        common_kwarg_dict = {"dim":layer_dim,
                            "depth":depths[i_layer],
                           "input_resolution":feature_hw,
                            "num_heads":num_heads[i_layer],
                            "window_size":window_sizes[i_layer],
                            "mlp_ratio":self.mlp_ratio, "qkv_bias":qkv_bias, 
                            "drop":drop_rate, "attn_drop":attn_drop_rate,
                            "drop_path":dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                            "norm_layer":self.model_norm_layer,
                            "act_layer":get_act(self.model_act_layer),
                            "pretrained_window_size":pretrained_window_sizes[i_layer],
                            "emb_dim_list":[],
                            "use_checkpoint":use_checkpoint[i_layer], "use_residual":use_residual
                            }

        self.mid_layer_1 = BasicLayerV2(**common_kwarg_dict)
        mid_attn_norm_layer = "instance" if self.norm_layer == "instance" else "rms"
        self.mid_attn = Attention(dim=layer_dim, num_heads=num_heads[i_layer],
                                  use_checkpoint=use_checkpoint[i_layer], norm_layer=mid_attn_norm_layer)
        self.mid_layer_2 = BasicLayerV2(**common_kwarg_dict)
        
        self.pool_layer = nn.Sequential(
                self.model_norm_layer(layer_dim),
                get_act(self.model_act_layer),
                AttentionPool1d(sequence_length=np.prod(feature_hw), embed_dim=layer_dim,
                                num_heads=8, output_dim=emb_chans, channel_first=False),
        )
        for bly in self.encode_layers:
            bly._init_respostnorm()
        self.mid_layer_1._init_respostnorm()
        self.mid_layer_2._init_respostnorm()
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
    
    def forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.init_layer(x)

        for encode_layer in self.encode_layers:
            x = encode_layer(x)

        x = self.mid_layer_1(x)
        x = self.mid_attn(x)
        x = self.mid_layer_2(x)
        x = self.pool_layer(x)
        return x
from timm.models.layers import trunc_normal_
import torch
import numpy as np
from functools import partial
from torch import nn
from torch.utils.checkpoint import checkpoint
from .swin_layers import Output2D
from .swin_layers_diffusion_2d import LinearAttention2D, LinearAttention, Attention2D, Attention, RMSNorm
from .swin_layers_diffusion_2d import BasicLayerV1, BasicLayerV2, SkipConv1D, SkipLinear, AttentionPool1d
from .swin_layers_diffusion_2d import default, prob_mask_like, LearnedSinusoidalPosEmb
from .swin_layers_diffusion_2d import PatchEmbed, PatchMerging, PatchMergingConv, PatchExpanding, ConvBlock2D
from .swin_layers_diffusion_2d import get_norm_layer_partial, get_norm_layer_partial_conv

from einops import rearrange, repeat

class SwinDiffusionUnet(nn.Module):
    def __init__(self, img_size=512, patch_size=4,
                 in_chans=1, cond_chans=3, out_chans=1, out_act=None,
                emb_chans=1024, num_class_embeds=None, cond_drop_prob=0.5,
                embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                window_sizes=[8, 4, 4, 2], mlp_ratio=4., qkv_bias=True, ape=True, patch_norm=True,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0, latent_drop_rate=0.0,
                use_checkpoint=False, pretrained_window_sizes=0,
                self_condition=False, use_residual=False):
        super().__init__()
        patch_size = int(patch_size)
        # for compability with Medsegdiff
        self.image_size = img_size
        self.input_img_channels = cond_chans
        self.mask_channels = in_chans
        
        ##################################
        self.patch_size = patch_size
        self.patch_norm = patch_norm
        self.out_chans = out_chans
        self.out_act = out_act
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
        self.use_residual = use_residual
        self.self_condition = self_condition
        ##################################
        self.num_layers = len(depths)
        self.ape = ape
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.feature_dim = int(self.embed_dim * 2 ** self.num_layers)

        if self.self_condition:
            in_chans = in_chans * 2
        
        time_emb_dim = embed_dim * 4
        sinu_pos_emb = LearnedSinusoidalPosEmb(time_emb_dim)
        fourier_dim = time_emb_dim + 1
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.latent_encoder = SwinDiffusionEncoder(img_size=img_size, patch_size=patch_size, in_chans=cond_chans, emb_chans=emb_chans,
                                                embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                                window_sizes=window_sizes, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, ape=ape,
                                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                                drop_path_rate=drop_path_rate, latent_drop_rate=latent_drop_rate,
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
                nn.SiLU(),
                nn.Linear(class_emb_dim, class_emb_dim)
            )
            emb_dim_list = [time_emb_dim, emb_chans, time_emb_dim]
        else:
            self.class_emb_layer = None
            self.null_class_emb = None
            self.class_mlp = None
            emb_dim_list = [time_emb_dim, emb_chans]
        self.emb_dim_list = emb_dim_list

        # stochastic depth
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                    in_chans=in_chans, embed_dim=embed_dim,
                                    norm_layer=RMSNorm if self.patch_norm else None)
        
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
        self.skip_layers, self.decode_layers = self.get_decode_layers()
        self.seg_final_layer, self.seg_final_expanding, self.out_conv = self.get_seg_final_layers()
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

    def get_layer_config_dict(self, dim, feature_resolution, i_layer):
        depths = self.depths
        common_kwarg_dict = {"dim":dim,
                            "input_resolution":feature_resolution,
                            "depth":self.depths[i_layer],
                            "num_heads":self.num_heads[i_layer],
                            "window_size":self.window_sizes[i_layer],
                            "mlp_ratio":self.mlp_ratio, "qkv_bias":self.qkv_bias, 
                            "drop":self.drop_rate, "attn_drop":self.attn_drop_rate,
                            "drop_path":self.dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                            "norm_layer":RMSNorm,
                            "pretrained_window_size":self.pretrained_window_sizes[i_layer],
                            "emb_dim_list":self.emb_dim_list,
                            "use_checkpoint":self.use_checkpoint[i_layer], "use_residual":self.use_residual
                            }
        return common_kwarg_dict
    
    def get_init_layer(self):
        common_kwarg_dict = self.get_layer_config_dict(self.embed_dim, self.patches_resolution, 0)
        init_layer = BasicLayerV2(**common_kwarg_dict)
        return init_layer
    
    def get_encode_layers(self):
        encode_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_dim = int(self.embed_dim * 2 ** i_layer)
            feature_resolution = np.array(self.patches_resolution) // (2 ** i_layer)
            common_kwarg_dict = self.get_layer_config_dict(layer_dim, feature_resolution, i_layer)
            common_kwarg_dict["downsample"] = PatchMerging
            encode_layer = BasicLayerV2(**common_kwarg_dict)
            encode_layers.append(encode_layer)
        return encode_layers
    
    def get_mid_layer(self):
        i_layer = self.num_layers - 1
        feature_dim = self.feature_dim
        common_kwarg_dict = self.get_layer_config_dict(feature_dim, self.feature_hw, i_layer)
        mid_layer_1 = BasicLayerV2(**common_kwarg_dict)
        mid_attn = Attention(dim=feature_dim, num_heads=self.num_heads[i_layer],
                                  use_checkpoint=self.use_checkpoint[i_layer])
        mid_layer_2 = BasicLayerV2(**common_kwarg_dict)
        return mid_layer_1, mid_attn, mid_layer_2
    
    def get_decode_layers(self):
        skip_layers = nn.ModuleList()
        decode_layers = nn.ModuleList()
        for d_i_layer in range(self.num_layers, 0, -1):
            i_layer = d_i_layer - 1
            layer_dim = int(self.embed_dim * 2 ** d_i_layer)
            feature_resolution = np.array(self.patches_resolution) // (2 ** d_i_layer)
            common_kwarg_dict = self.get_layer_config_dict(layer_dim, feature_resolution, i_layer)
            common_kwarg_dict["upsample"] = PatchExpanding
            skip_layer = SkipLinear(layer_dim * 2, layer_dim,
                                    norm=RMSNorm)
            decode_layer = BasicLayerV1(**common_kwarg_dict)
            
            skip_layers.append(skip_layer)
            decode_layers.append(decode_layer)

        return skip_layers, decode_layers

    def get_seg_final_layers(self):
        
        i_layer = 0
        common_kwarg_dict = self.get_layer_config_dict(self.embed_dim, self.patches_resolution, i_layer)
        seg_final_layer = BasicLayerV1(**common_kwarg_dict)
        seg_final_expanding = PatchExpanding(input_resolution=self.patches_resolution,
                                                    dim=self.embed_dim,
                                                    return_vector=False,
                                                    dim_scale=self.patch_size,
                                                    norm_layer=RMSNorm
                                                    )
        out_conv = Output2D(self.embed_dim // 2, self.out_chans, act=self.out_act)
        return seg_final_layer, seg_final_expanding, out_conv
    
class SwinDiffusionEncoder(SwinDiffusionUnet):
    def __init__(self, img_size=512, patch_size=4, in_chans=1, emb_chans=1024,
                embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                window_sizes=[8, 4, 4, 2], mlp_ratio=4., qkv_bias=True, ape=True, patch_norm=True,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, latent_drop_rate=0.1,
                use_checkpoint=False, pretrained_window_sizes=0,
                use_residual=False
                ):
        super(SwinDiffusionUnet, self).__init__()
        patch_size = int(patch_size)

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
        if isinstance(pretrained_window_sizes, int):
            pretrained_window_sizes = [pretrained_window_sizes for _ in num_heads]
        self.pretrained_window_sizes = pretrained_window_sizes
        if isinstance(use_checkpoint, bool):
            use_checkpoint = [use_checkpoint for _ in num_heads]
        self.use_checkpoint = use_checkpoint
        self.use_residual = use_residual
        ##################################
        self.num_layers = len(depths)
        self.ape = ape
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        feature_dim = int(self.embed_dim * 2 ** self.num_layers)
        self.feature_dim = feature_dim
        self.emb_dim_list = []
        # stochastic depth
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                    in_chans=in_chans, embed_dim=embed_dim,
                                    norm_layer=RMSNorm)
        
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
                nn.SiLU(),
                AttentionPool1d(sequence_length=np.prod(self.feature_hw), embed_dim=feature_dim,
                                num_heads=8, output_dim=feature_dim * 2, channel_first=False),
                nn.Linear(feature_dim * 2, feature_dim),
                nn.Dropout(p=latent_drop_rate),
                nn.SiLU(),
                nn.Linear(feature_dim, emb_chans),
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
    

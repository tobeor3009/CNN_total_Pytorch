from timm.models.layers import trunc_normal_
import torch
import numpy as np
from functools import partial
from torch import nn
from copy import deepcopy
from .swin_layers import Output2D
from .swin_layers_diffusion_2d import BasicLayerV1, BasicLayerV2, SkipConv1D, AttentionPool1d
from .swin_layers_diffusion_2d import exists, default, extract, SinusoidalPosEmb
from .swin_layers_diffusion_2d import PatchEmbed, PatchMergingConv, PatchExpandingMulti


class GroupNormChannelFirst(nn.GroupNorm):
    
    def __init__(self, num_channels, *args, **kwargs):
        super().__init__(num_channels=num_channels,*args, **kwargs)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        return super().forward(x).permute(0, 2, 1)

def get_norm_layer_partial(num_groups):
    return partial(GroupNormChannelFirst, num_groups=num_groups)
class SwinDiffusionUnet(nn.Module):
    def __init__(self, img_size=512, patch_size=4,
                 in_chans=1, cond_chans=3, out_chans=1, out_act=None,
                emb_chans=1024, num_class_embeds=None,
                embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                window_sizes=[8, 4, 4, 2], mlp_ratio=4., qkv_bias=True, ape=True,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                patch_norm=True, skip_connect=True,
                use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0],
                self_condition=False, use_residual=False
                ):
        super().__init__()
        patch_size = int(patch_size)
        # for compability with Medsegdiff 
        self.image_size = img_size
        self.input_img_channels = cond_chans
        self.mask_channels = in_chans
        
        ##################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.skip_connect = skip_connect
        self.self_condition = self_condition
        if self.self_condition:
            in_chans = in_chans * 2

        time_emb_dim = embed_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim // 2),
            nn.Linear(time_emb_dim // 2, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.latent_encoder = SwinDiffusionEncoder(img_size=img_size, patch_size=patch_size, in_chans=cond_chans, emb_chans=emb_chans,
                                                embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                                window_sizes=window_sizes, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, ape=ape,
                                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                                                use_checkpoint=use_checkpoint, pretrained_window_sizes=pretrained_window_sizes,
                                                patch_norm=patch_norm, use_residual=use_residual
                                                )


        # class embedding

        emb_dim_list = [time_emb_dim, emb_chans]
        self.num_class_embeds = num_class_embeds
        self.r_class_emb_layer = nn.Embedding(num_class_embeds, time_emb_dim)
        self.g_class_emb_layer = nn.Embedding(num_class_embeds, time_emb_dim)
        self.b_class_emb_layer = nn.Embedding(num_class_embeds, time_emb_dim)
        self.rgb_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim * 3, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        emb_dim_list += [time_emb_dim]

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                    in_chans=in_chans, embed_dim=embed_dim,
                                    norm_layer=get_norm_layer_partial(num_heads[0]) if self.patch_norm else None)
        
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            pos_embed_shape = torch.zeros(1, num_patches, embed_dim)
            self.absolute_pos_embed = nn.Parameter(pos_embed_shape)
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # build layers
        self.encode_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_dim = int(embed_dim * 2 ** i_layer)
            feature_resolution = np.array((patches_resolution[0] // (2 ** i_layer),
                                            patches_resolution[1] // (2 ** i_layer)))
            encode_layer = BasicLayerV1(dim=layer_dim,
                                        input_resolution=feature_resolution,
                                        depth=depths[i_layer],
                                        num_heads=num_heads[i_layer],
                                        window_size=window_sizes[i_layer],
                                        mlp_ratio=self.mlp_ratio,
                                        qkv_bias=qkv_bias,
                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=dpr[sum(depths[:i_layer]):sum(
                                            depths[:i_layer + 1])],
                                        norm_layer=get_norm_layer_partial(num_heads[i_layer]),
                                        downsample=PatchMergingConv if (i_layer < self.num_layers - 1) else None,
                                        use_checkpoint=use_checkpoint,
                                        pretrained_window_size=pretrained_window_sizes[i_layer],
                                        emb_dim_list=emb_dim_list, use_residual=use_residual)
            self.encode_layers.append(encode_layer)
        depth_level = self.num_layers - 1
        feature_hw = (patches_resolution[0] // (2 ** depth_level),
                    patches_resolution[1] // (2 ** depth_level))
        
        self.mid_layer = BasicLayerV2(dim=layer_dim,
                                    input_resolution=feature_hw,
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    window_size=window_sizes[i_layer],
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(
                                        depths[:i_layer + 1])],
                                    norm_layer=get_norm_layer_partial(num_heads[i_layer]),
                                    upsample=None,
                                    use_checkpoint=use_checkpoint,
                                    pretrained_window_size=pretrained_window_sizes[i_layer],
                                    emb_dim_list=emb_dim_list, use_residual=use_residual)
        self.skip_conv_layers = nn.ModuleList()
        self.decode_layers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1, -1, -1):
            layer_dim = int(embed_dim * 2 ** i_layer)
            feature_resolution = np.array((patches_resolution[0] // (2 ** i_layer),
                                            patches_resolution[1] // (2 ** i_layer)))
            skip_conv_layer = SkipConv1D(layer_dim * 2, layer_dim)

            decode_layer = BasicLayerV2(dim=layer_dim,
                                    input_resolution=feature_resolution,
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    window_size=window_sizes[i_layer],
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(
                                        depths[:i_layer + 1])],
                                    norm_layer=get_norm_layer_partial(num_heads[i_layer]),
                                    upsample=PatchExpandingMulti if (i_layer > 0) else None,
                                    use_checkpoint=use_checkpoint,
                                    pretrained_window_size=pretrained_window_sizes[i_layer],
                                    emb_dim_list=emb_dim_list, use_residual=use_residual)
            self.skip_conv_layers.append(skip_conv_layer)
            self.decode_layers.append(decode_layer)
        self.seg_final_expanding = PatchExpandingMulti(input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                                         patches_resolution[1] // (2 ** i_layer)),
                                                        dim=layer_dim,
                                                        return_vector=False,
                                                        dim_scale=patch_size,
                                                        norm_layer=get_norm_layer_partial(num_heads[i_layer])
                                                        )
        self.seg_final_conv = Output2D(layer_dim // 2, out_chans, act=out_act)
        for bly in self.encode_layers:
            bly._init_respostnorm()
        self.mid_layer._init_respostnorm()
        for bly in self.decode_layers:
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

    def forward(self, x, time, cond=None, x_self_cond=None, class_labels=None):

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        time_emb = self.time_mlp(time)
        latent = self.latent_encoder(cond)

        r_class, g_class, b_class = class_labels.chunk(3, dim=1)
        r_class, g_class, b_class = r_class.squeeze(1), g_class.squeeze(1), b_class.squeeze(1)
        r_class = self.r_class_emb_layer(r_class).to(dtype=x.dtype)
        g_class = self.g_class_emb_layer(g_class).to(dtype=x.dtype)
        b_class = self.b_class_emb_layer(b_class).to(dtype=x.dtype)
        class_emb = torch.cat([r_class, g_class, b_class], dim=1)
        class_emb = self.rgb_mlp(class_emb)

        emb_list = [time_emb, latent, class_emb]

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        skip_connect_list = []
        for idx, encode_layer in enumerate(self.encode_layers):
            x = encode_layer(x, *emb_list)
            if idx < len(self.encode_layers) - 1:
                skip_connect_list.insert(0, x)

        x = self.mid_layer(x, *emb_list)

        for idx, (skip_conv_layer, decode_layer) in enumerate(zip(self.skip_conv_layers, self.decode_layers)):
            if idx < len(self.decode_layers) - 1 and self.skip_connect:
                skip_x = skip_connect_list[idx]
                x = skip_conv_layer(x, skip_x)
            x = decode_layer(x, *emb_list)

        x = self.seg_final_expanding(x)
        x = self.seg_final_conv(x)
        return x

    def print_tensor_info(self, tensor):
        print(tensor.min(), tensor.max(), torch.isnan(tensor).any())



class SwinDiffusionEncoder(nn.Module):
    def __init__(self, img_size=512, patch_size=4, in_chans=1, emb_chans=1024,
                embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                window_sizes=[8, 4, 4, 2], mlp_ratio=4., qkv_bias=True, ape=True,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0],
                patch_norm=True, use_residual=False
                ):
        super().__init__()
        patch_size = int(patch_size)

        ##################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                    in_chans=in_chans, embed_dim=embed_dim,
                                    norm_layer=get_norm_layer_partial(num_heads[0]) if self.patch_norm else None)
        
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            pos_embed_shape = torch.zeros(1, num_patches, embed_dim)
            self.absolute_pos_embed = nn.Parameter(pos_embed_shape)
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # build layers
        self.encode_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_dim = int(embed_dim * 2 ** i_layer)
            feature_resolution = np.array((patches_resolution[0] // (2 ** i_layer),
                                            patches_resolution[1] // (2 ** i_layer)))
            encode_layer = BasicLayerV1(dim=layer_dim,
                                        input_resolution=feature_resolution,
                                        depth=depths[i_layer],
                                        num_heads=num_heads[i_layer],
                                        window_size=window_sizes[i_layer],
                                        mlp_ratio=self.mlp_ratio,
                                        qkv_bias=qkv_bias,
                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=dpr[sum(depths[:i_layer]):sum(
                                            depths[:i_layer + 1])],
                                        norm_layer=get_norm_layer_partial(num_heads[i_layer]),
                                        downsample=PatchMergingConv if (i_layer < self.num_layers - 1) else None,
                                        use_checkpoint=use_checkpoint,
                                        pretrained_window_size=pretrained_window_sizes[i_layer],
                                        emb_dim_list=[], use_residual=use_residual)
            self.encode_layers.append(encode_layer)
        depth_level = self.num_layers - 1
        feature_hw = (patches_resolution[0] // (2 ** depth_level),
                    patches_resolution[1] // (2 ** depth_level))
        
        self.mid_layer = BasicLayerV2(dim=layer_dim,
                                    input_resolution=feature_hw,
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    window_size=window_sizes[i_layer],
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(
                                        depths[:i_layer + 1])],
                                    norm_layer=get_norm_layer_partial(num_heads[i_layer]),
                                    upsample=None,
                                    use_checkpoint=use_checkpoint,
                                    pretrained_window_size=pretrained_window_sizes[i_layer],
                                    emb_dim_list=[], use_residual=use_residual)
        
        self.pool_layer = AttentionPool1d(sequence_length=np.prod(feature_hw), embed_dim=layer_dim,
                                          num_heads=8, output_dim=emb_chans, channel_first=False)
        for bly in self.encode_layers:
            bly._init_respostnorm()
        self.mid_layer._init_respostnorm()
            
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

        for encode_layer in self.encode_layers:
            x = encode_layer(x)

        x = self.mid_layer(x)
        x = self.pool_layer(x)
        return x
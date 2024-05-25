from timm.models.layers import trunc_normal_
import torch
import numpy as np
from functools import partial
from torch import nn
from copy import deepcopy
from .swin_layers import Output2D
from .swin_layers_diffusion_2d import BasicLayerV1, BasicLayerV2, SkipConv1D, AttentionPool1d
from .swin_layers_diffusion_2d import exists, default, extract, SinusoidalPosEmb
from .swin_layers_diffusion_2d import PatchEmbed, PatchMergingConv, PatchExpandingMulti, Attention, ConvBlock2D
from .swin_layers_diffusion_2d import GroupNormChannelFirst, WrapGroupNorm



def get_norm_layer_partial(num_groups):
    return partial(GroupNormChannelFirst, num_groups=num_groups)

def get_norm_layer_partial_conv(num_groups):
    return partial(WrapGroupNorm, num_groups=num_groups)

class SwinDiffusionUnet(nn.Module):
    def __init__(self, img_size=512, patch_size=4,
                 in_chans=1, cond_chans=3, out_chans=1, out_act=None,
                emb_chans=1024, num_class_embeds=None,
                embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                window_sizes=[8, 4, 4, 2], mlp_ratio=4., qkv_bias=True, ape=True,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                patch_norm=True,
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
        if num_class_embeds is not None:
            self.class_emb_layer = nn.Embedding(num_class_embeds, time_emb_dim * 2)
            self.class_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim * 2, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim)
            )
            emb_dim_list += [time_emb_dim]
        else:
            self.class_emb_layer = None
            self.class_mlp = None
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
        self.encode_layers_1 = nn.ModuleList()
        self.encode_layers_2 = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_dim = int(embed_dim * 2 ** i_layer)
            feature_resolution = np.array((patches_resolution[0] // (2 ** i_layer),
                                            patches_resolution[1] // (2 ** i_layer)))
            
            encode_layer_1 = BasicLayerV1(dim=layer_dim,
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
                                        downsample=PatchMergingConv,
                                        use_checkpoint=use_checkpoint,
                                        pretrained_window_size=pretrained_window_sizes[i_layer],
                                        emb_dim_list=emb_dim_list, use_residual=use_residual)
            
            encode_layer_2 = BasicLayerV1(dim=layer_dim * 2,
                                        input_resolution=feature_resolution // 2,
                                        depth=depths[i_layer],
                                        num_heads=num_heads[i_layer],
                                        window_size=window_sizes[i_layer],
                                        mlp_ratio=self.mlp_ratio,
                                        qkv_bias=qkv_bias,
                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=dpr[sum(depths[:i_layer]):sum(
                                            depths[:i_layer + 1])],
                                        norm_layer=get_norm_layer_partial(num_heads[i_layer]),
                                        downsample=None,
                                        use_checkpoint=use_checkpoint,
                                        pretrained_window_size=pretrained_window_sizes[i_layer],
                                        emb_dim_list=emb_dim_list, use_residual=use_residual)
            
            self.encode_layers_1.append(encode_layer_1)
            self.encode_layers_2.append(encode_layer_2)

        depth_level = self.num_layers
        layer_dim = int(embed_dim * 2 ** depth_level)
        feature_hw = (patches_resolution[0] // (2 ** depth_level),
                    patches_resolution[1] // (2 ** depth_level))
        
        self.mid_attn = Attention(dim=layer_dim, num_heads=num_heads[i_layer])
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
        self.skip_conv_layers_1 = nn.ModuleList()
        self.skip_conv_layers_2 = nn.ModuleList()
        self.decode_layers_1 = nn.ModuleList()
        self.decode_layers_2 = nn.ModuleList()
        for d_i_layer in range(self.num_layers, 0, -1):

            i_layer = d_i_layer - 1
            layer_dim = int(embed_dim * 2 ** d_i_layer)
            feature_resolution = np.array((patches_resolution[0] // (2 ** d_i_layer),
                                            patches_resolution[1] // (2 ** d_i_layer)))

            skip_conv_layer_1 = SkipConv1D(layer_dim * 2, layer_dim)
            skip_conv_layer_2 = SkipConv1D(layer_dim * 2, layer_dim)

            decode_layer_1 = BasicLayerV2(dim=layer_dim,
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
                                    upsample=None,
                                    use_checkpoint=use_checkpoint,
                                    pretrained_window_size=pretrained_window_sizes[i_layer],
                                    emb_dim_list=emb_dim_list, use_residual=use_residual)
            
            decode_layer_2 = BasicLayerV2(dim=layer_dim,
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
                                    upsample=PatchExpandingMulti,
                                    use_checkpoint=use_checkpoint,
                                    pretrained_window_size=pretrained_window_sizes[i_layer],
                                    emb_dim_list=emb_dim_list, use_residual=use_residual)

            self.skip_conv_layers_1.append(skip_conv_layer_1)
            self.skip_conv_layers_2.append(skip_conv_layer_2)
            self.decode_layers_1.append(decode_layer_1)
            self.decode_layers_2.append(decode_layer_2)

        self.seg_final_layer = BasicLayerV2(dim=embed_dim,
                                            input_resolution=patches_resolution,
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
        
        self.seg_final_expanding = PatchExpandingMulti(input_resolution=(patches_resolution[0],
                                                                         patches_resolution[1]),
                                                        dim=embed_dim,
                                                        return_vector=False,
                                                        dim_scale=patch_size,
                                                        norm_layer=get_norm_layer_partial(num_heads[i_layer])
                                                        )
        
        emb_type_list = ["seq", "seq", "seq"]
        self.seg_conv_1 = ConvBlock2D(embed_dim // 2, embed_dim, 3,
                                    stride=1, norm=get_norm_layer_partial_conv(num_heads[i_layer]), bias=False,
                                    emb_dim_list=emb_dim_list, emb_type_list=emb_type_list,
                                    attn_info=None, use_checkpoint=use_checkpoint)
        self.seg_conv_2 = ConvBlock2D(embed_dim, embed_dim // 2, 3,
                                    stride=1, norm=get_norm_layer_partial_conv(num_heads[i_layer]), bias=False,
                                    emb_dim_list=emb_dim_list, emb_type_list=emb_type_list,
                                    attn_info=None, use_checkpoint=use_checkpoint)
        self.seg_final_conv = Output2D(embed_dim // 2, out_chans, act=out_act)

        for bly in (self.encode_layers_1 + self.encode_layers_2):
            bly._init_respostnorm()
        self.mid_layer._init_respostnorm()
        for bly in (self.decode_layers_1 + self.decode_layers_2):
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

        if class_labels is not None:
            class_emb = self.class_emb_layer(class_labels).to(dtype=x.dtype)
            class_emb = self.class_mlp(class_emb)
            emb_list = [time_emb, latent, class_emb]
        else:
            emb_list = [time_emb, latent]

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        skip_connect_list_1 = []
        skip_connect_list_2 = []
        for encode_layer_1, encode_layer_2 in zip(self.encode_layers_1, self.encode_layers_2):
            
            x = encode_layer_1(x, *emb_list)
            skip_connect_list_1.append(x)
            
            x = encode_layer_2(x, *emb_list)
            skip_connect_list_2.append(x)

        x = self.mid_attn(x)
        x = self.mid_layer(x, *emb_list)

        for skip_conv_layer_1, skip_conv_layer_2, decode_layer_1, decode_layer_2 in zip(self.skip_conv_layers_1, self.skip_conv_layers_2,
                                                                                                      self.decode_layers_1, self.decode_layers_2):
            
            skip_x = skip_connect_list_1.pop()
            x = skip_conv_layer_1(x, skip_x)
            x = decode_layer_1(x, *emb_list)

            skip_x = skip_connect_list_2.pop()
            x = skip_conv_layer_2(x, skip_x)
            x = decode_layer_2(x, *emb_list)

        x = self.seg_final_layer(x, *emb_list)
        x = self.seg_final_expanding(x)
        
        x = self.seg_conv_1(x, *emb_list)
        x = self.seg_conv_2(x, *emb_list)
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
        self.encode_layers_1 = nn.ModuleList()
        self.encode_layers_2 = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_dim = int(embed_dim * 2 ** i_layer)
            feature_resolution = np.array((patches_resolution[0] // (2 ** i_layer),
                                            patches_resolution[1] // (2 ** i_layer)))
            
            encode_layer_1 = BasicLayerV1(dim=layer_dim,
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
                                        downsample=PatchMergingConv,
                                        use_checkpoint=use_checkpoint,
                                        pretrained_window_size=pretrained_window_sizes[i_layer],
                                        emb_dim_list=[], use_residual=use_residual)
            
            encode_layer_2 = BasicLayerV1(dim=layer_dim * 2,
                                        input_resolution=feature_resolution // 2,
                                        depth=depths[i_layer],
                                        num_heads=num_heads[i_layer],
                                        window_size=window_sizes[i_layer],
                                        mlp_ratio=self.mlp_ratio,
                                        qkv_bias=qkv_bias,
                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=dpr[sum(depths[:i_layer]):sum(
                                            depths[:i_layer + 1])],
                                        norm_layer=get_norm_layer_partial(num_heads[i_layer]),
                                        downsample=None,
                                        use_checkpoint=use_checkpoint,
                                        pretrained_window_size=pretrained_window_sizes[i_layer],
                                        emb_dim_list=[], use_residual=use_residual)
            
            self.encode_layers_1.append(encode_layer_1)
            self.encode_layers_2.append(encode_layer_2)

        depth_level = self.num_layers
        layer_dim = int(embed_dim * 2 ** depth_level)
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
        
        self.pool_layer = nn.Sequential(
                get_norm_layer_partial(num_heads[i_layer])(layer_dim),
                nn.SiLU(),
                AttentionPool1d(sequence_length=np.prod(feature_hw), embed_dim=layer_dim,
                                num_heads=8, output_dim=emb_chans, channel_first=False),
        )
        for bly in (self.encode_layers_1 + self.encode_layers_2):
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

        for encode_layer_1, encode_layer_2 in zip(self.encode_layers_1, self.encode_layers_2):
            x = encode_layer_1(x)
            x = encode_layer_2(x)

        x = self.mid_layer(x)
        x = self.pool_layer(x)
        return x
import torch
import numpy as np
from torch import nn
from .swin_layers import PatchEmbed, BasicLayer, PatchMerging, PatchExpanding
from .swin_layers import trunc_normal_
from ..layers import ConvBlock3D
from ..layers import get_act


class SwinTransformerMultiTask(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    def __init__(self, img_size=512, patch_size=4, in_chans=3,
                 num_classes=1000, seg_num_classes=10, validity_shape=(4, 4, 4),
                 embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_sizes=[8, 4, 4, 2], mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 class_act="softmax", seg_act="sigmoid", validity_act="sigmoid",
                 get_class=True, get_seg=False, get_validity=False,
                 norm_layer=nn.LayerNorm, ape=True, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0],
                 **kwargs):
        super().__init__()

        patch_size = int(patch_size)

        self.num_classes = num_classes
        self.seg_num_classes = seg_num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.get_class = get_class
        self.get_seg = get_seg
        self.get_validity = get_validity

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim,
                                      norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            pos_embed_shape = torch.zeros(1, num_patches, embed_dim)
            self.absolute_pos_embed = nn.Parameter(pos_embed_shape)
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        # build layers
        self.encode_layers = nn.ModuleList()
        i_layer = 0
        first_encode_layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                        input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                          patches_resolution[1] // (
                                                              2 ** i_layer),
                                                          patches_resolution[2] // (2 ** i_layer)),
                                        depth=depths[i_layer],
                                        num_heads=num_heads[i_layer],
                                        window_size=window_sizes[i_layer],
                                        mlp_ratio=self.mlp_ratio,
                                        qkv_bias=qkv_bias,
                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=dpr[sum(depths[:i_layer]):sum(
                                            depths[:i_layer + 1])],
                                        norm_layer=norm_layer,
                                        downsample=None,
                                        use_checkpoint=use_checkpoint,
                                        pretrained_window_size=pretrained_window_sizes[i_layer])
        self.encode_layers.append(first_encode_layer)
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (
                                                     2 ** i_layer),
                                                 patches_resolution[2] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_sizes[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(
                                   depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (
                                   i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.encode_layers.append(layer)

        if get_class:
            self.class_head = SwinClassificationHead(self.num_features, num_classes,
                                                     class_act=class_act, norm_layer=norm_layer)
        if get_seg:
            self.concat_linears = nn.ModuleList()
            self.decode_layers = nn.ModuleList()
            for i_layer in range(self.num_layers - 1, -1, -1):
                target_dim = int(embed_dim * 2 ** i_layer)
                if i_layer > 0:
                    # concat_linear = nn.Conv1d(target_dim * 2, target_dim,
                    #                           kernel_size=1, bias=False)
                    concat_linear = nn.Linear(target_dim * 2, target_dim,
                                              bias=False)
                else:
                    concat_linear = nn.Identity()
                layer = BasicLayer(dim=target_dim,
                                   input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                     patches_resolution[1] // (
                                                         2 ** i_layer),
                                                     patches_resolution[2] // (2 ** i_layer)),
                                   depth=depths[i_layer],
                                   num_heads=num_heads[i_layer],
                                   window_size=window_sizes[i_layer],
                                   mlp_ratio=self.mlp_ratio,
                                   qkv_bias=qkv_bias,
                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                   drop_path=dpr[sum(depths[:i_layer]):sum(
                                       depths[:i_layer + 1])],
                                   norm_layer=norm_layer,
                                   upsample=PatchExpanding if (
                                       i_layer > 0) else None,
                                   use_checkpoint=use_checkpoint,
                                   pretrained_window_size=pretrained_window_sizes[i_layer])
                self.concat_linears.append(concat_linear)
                self.decode_layers.append(layer)
            for bly in self.decode_layers:
                bly._init_respostnorm()
            self.seg_final_expanding = PatchExpanding(input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                                        patches_resolution[1] // (
                                                                            2 ** i_layer),
                                                                        patches_resolution[2] // (2 ** i_layer)),
                                                      dim=target_dim,
                                                      return_vector=False,
                                                      dim_scale=patch_size,
                                                      norm_layer=norm_layer
                                                      )
            self.seg_final_conv = nn.Conv2d(target_dim // 2, seg_num_classes,
                                            kernel_size=1, padding=0)
            self.seg_final_act = get_act(seg_act)
        if get_validity:
            depth_level = self.num_layers - 1
            self.validity_dim = int(embed_dim * (2 ** depth_level))
            self.validity_zhw = (patches_resolution[0] // (2 ** depth_level),
                                 patches_resolution[1] // (2 ** depth_level),
                                 patches_resolution[2] // (2 ** depth_level))
            self.validity_conv_1 = ConvBlock3D(self.validity_dim, embed_dim,
                                               kernel_size=3, act="gelu")
            self.validity_avg_pool = nn.AdaptiveAvgPool3d(validity_shape)
            self.validity_conv_2 = ConvBlock3D(embed_dim, embed_dim,
                                               kernel_size=3, act="gelu")
            self.validity_out_conv = ConvBlock3D(embed_dim, in_chans,
                                                 kernel_size=1, act=validity_act)

        self.apply(self._init_weights)
        for bly in self.encode_layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def encode_forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        skip_connect_list = []
        for idx, layer in enumerate(self.encode_layers):
            x = layer(x)
            if idx < len(self.encode_layers) - 1 and idx > 0:
                skip_connect_list.insert(0, x)
        return x, skip_connect_list

    def decode_forward(self, x, skip_connect_list):
        for idx, (concat_linear, layer) in enumerate(zip(self.concat_linears,
                                                         self.decode_layers)):
            if idx < len(self.decode_layers) - 1:
                skip_connect = skip_connect_list[idx]
                x = torch.cat([x, skip_connect], dim=-1)
                # x = concat_linear(x.permute(0, 2, 1)).permute(0, 2, 1)
                x = concat_linear(x)
            x = layer(x)
        x = self.seg_final_expanding(x)
        x = self.seg_final_conv(x)
        x = self.seg_final_act(x)
        return x

    def validity_forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.view(-1, self.validity_dim,
                   *self.validity_zhw)
        x = self.validity_conv_1(x)
        x = self.validity_avg_pool(x)
        x = self.validity_conv_2(x)
        x = self.validity_out_conv(x)
        return x

    def forward(self, x):
        output = []
        x, skip_connect_list = self.encode_forward(x)
        if self.get_class:
            class_output = self.class_head(x)
            output.append(class_output)
        if self.get_seg:
            seg_output = self.decode_forward(x, skip_connect_list)
            output.append(seg_output)
        if self.get_validity:
            validity_output = self.validity_forward(x)
            output.append(validity_output)
        if len(output) == 1:
            output = output[0]
        return output

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * \
            np.prod(self.patches_resolution) // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


class SwinClassificationHead(nn.Module):
    def __init__(self, input_feature, num_classes, class_act, norm_layer):
        super().__init__()
        self.norm = norm_layer(input_feature)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(input_feature,
                                num_classes) if num_classes > 0 else nn.Identity()
        self.act = get_act(class_act)

    def forward(self, x):
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)  # B C
        x = self.linear(x)
        x = self.act(x)
        return x

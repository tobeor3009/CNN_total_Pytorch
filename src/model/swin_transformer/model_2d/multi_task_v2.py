import torch
import numpy as np
from torch import nn
from .swin_layers import PatchEmbed, PatchMerging, PatchExpanding
from .swin_layers import BasicLayerV1, BasicLayerV2, Output2D
from .swin_layers import trunc_normal_, to_2tuple
from ..layers import ConvBlock2D, AttentionPool1d
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
                 num_classes=1000, seg_num_classes=10, validity_shape=(1, 8, 8), inject_num_classes=None,
                 embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_sizes=[8, 4, 4, 2], mlp_ratio=4., qkv_bias=True, ape=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 class_act="softmax", seg_act="sigmoid", validity_act="sigmoid",
                 get_class=True, get_seg=False, get_validity=False,
                 norm_layer=nn.LayerNorm, patch_norm=True, skip_connect=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0]):
        super().__init__()

        patch_size = int(patch_size)

        self.num_classes = num_classes
        self.seg_num_classes = seg_num_classes
        self.inject_num_classes = inject_num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.get_class = get_class
        self.get_seg = get_seg
        self.get_validity = get_validity
        self.skip_connect = skip_connect
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim,
                                      norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

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
        for i_layer in range(self.num_layers):
            BasicLayer = BasicLayerV1 if i_layer == 0 else BasicLayerV2
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
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
        depth_level = self.num_layers - 1
        feature_hw = (patches_resolution[0] // (2 ** depth_level),
                      patches_resolution[1] // (2 ** depth_level))

        if get_class:
            self.class_head = SwinClassificationHead(np.prod(feature_hw), self.num_features,
                                                     num_classes, class_act=class_act)
        if get_seg:
            self.cat_linears = nn.ModuleList()
            self.decode_layers = nn.ModuleList()
            for i_layer in range(self.num_layers - 1, -1, -1):
                target_dim = int(embed_dim * 2 ** i_layer)
                if i_layer > 0 and skip_connect:
                    cat_linear = nn.Linear(target_dim * 2, target_dim,
                                           bias=False)
                else:
                    cat_linear = nn.Identity()
                layer = BasicLayerV2(dim=target_dim,
                                     input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                       patches_resolution[1] // (2 ** i_layer)),
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
                self.cat_linears.append(cat_linear)
                self.decode_layers.append(layer)
            for bly in self.decode_layers:
                bly._init_respostnorm()
            self.seg_final_expanding = PatchExpanding(input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                                        patches_resolution[1] // (2 ** i_layer)),
                                                      dim=target_dim,
                                                      return_vector=False,
                                                      dim_scale=patch_size,
                                                      norm_layer=norm_layer
                                                      )
            self.seg_final_conv = Output2D(target_dim // 2, seg_num_classes,
                                            act=seg_act)
        if get_validity:
            self.validity_dim = int(embed_dim * (2 ** depth_level))
            self.validity_hw = feature_hw
            self.validity_conv_1 = ConvBlock2D(self.validity_dim, embed_dim,
                                               kernel_size=3, act="gelu", norm=None)
            self.validity_avg_pool = nn.AdaptiveAvgPool2d(validity_shape[1:])
            self.validity_out_conv = ConvBlock2D(embed_dim, validity_shape[0],
                                                 kernel_size=1, act=validity_act, norm=None)

        if inject_num_classes is not None:
            feature_channel = int(embed_dim * (2 ** depth_level))
            self.inject_linear = nn.Linear(inject_num_classes,
                                           feature_channel, bias=False)
            self.inject_norm = norm_layer(feature_channel)
            inject_pos_embed_shape = torch.zeros(1,
                                                 np.prod(feature_hw),
                                                 1)
            self.inject_absolute_pos_embed = nn.Parameter(
                inject_pos_embed_shape)
            trunc_normal_(self.inject_absolute_pos_embed, std=.02)
            self.inject_cat_linear = nn.Linear(feature_channel * 2,
                                               feature_channel, bias=False)
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
            if idx < len(self.encode_layers) - 1:
                skip_connect_list.insert(0, x)
        return x, skip_connect_list

    def decode_forward(self, x, skip_connect_list):
        for idx, (cat_linear, layer) in enumerate(zip(self.cat_linears,
                                                      self.decode_layers)):
            if idx < len(self.decode_layers) - 1 and self.skip_connect:
                skip_connect = skip_connect_list[idx]
                x = torch.cat([x, skip_connect], dim=-1)
                x = cat_linear(x)
            x = layer(x)
        x = self.seg_final_expanding(x)
        x = self.seg_final_conv(x)
        return x

    def validity_forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.view(-1, self.validity_dim,
                   *self.validity_hw)
        x = self.validity_conv_1(x)
        x = self.validity_avg_pool(x)
        x = self.validity_out_conv(x)
        return x

    def forward(self, x, inject_class=None):
        output = []
        x, skip_connect_list = self.encode_forward(x)
        if self.inject_num_classes is not None:
            inject_class = self.inject_linear(inject_class)
            inject_class = self.inject_norm(inject_class)
            inject_class = inject_class.unsqueeze(1).repeat(1, x.shape[1], 1)
            inject_class = inject_class + self.inject_absolute_pos_embed
            x = torch.cat([x, inject_class], dim=-1)
            x = self.inject_cat_linear(x)
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
        if len(output) == 0:
            output = x
        return output

    def forward_debug(self, x):
        output = []
        x, skip_connect_list = self.encode_forward(x)
        self.print_tensor_info(x)
        if self.inject_num_classes is not None:
            inject_class = self.inject_linear(inject_class)
            inject_class = self.inject_norm(inject_class)
            inject_class = inject_class.unsqueeze(1).repeat(1, x.shape[1], 1)
            inject_class = inject_class + self.inject_absolute_pos_embed
            x = torch.cat([x, inject_class], dim=-1)
            x = self.inject_cat_linear(x)
        if self.get_class:
            class_output = self.class_head(x)
            self.print_tensor_info(class_output)
            output.append(class_output)
        if self.get_seg:
            seg_output = self.decode_forward(x, skip_connect_list)
            self.print_tensor_info(seg_output)
            output.append(seg_output)
            self.print_tensor_info(x)
        if self.get_validity:
            validity_output = self.validity_forward(x)
            self.print_tensor_info(validity_output)
            output.append(validity_output)
        if len(output) == 1:
            output = output[0]
        if len(output) == 0:
            output = x
        return output

    def encode_forward_bebug(self, x):
        x = self.patch_embed(x)
        self.print_tensor_info(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        self.print_tensor_info(x)
        x = self.pos_drop(x)
        self.print_tensor_info(x)
        skip_connect_list = []
        for idx, layer in enumerate(self.encode_layers):
            x = layer(x)
            self.print_tensor_info(x)
            if idx < len(self.encode_layers) - 1:
                skip_connect_list.insert(0, x)

        return x, skip_connect_list

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * \
            self.patches_resolution[0] * \
            self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

    def print_tensor_info(self, tensor):
        print(tensor.min(), tensor.max(), torch.isnan(tensor).any())


class SwinClassificationHead(nn.Module):
    def __init__(self, patch_num, input_feature, num_classes, class_act,
                 dropout_proba=0.05):
        super().__init__()
        self.attn_pool = AttentionPool1d(patch_num, input_feature,
                                         num_heads=4, output_dim=input_feature * 2,
                                         channel_first=False)
        self.dropout = nn.Dropout(p=dropout_proba, inplace=True)
        self.linear = nn.Linear(input_feature * 2,
                                num_classes) if num_classes > 0 else nn.Identity()
        self.act = get_act(class_act)

    def forward(self, x):
        x = self.attn_pool(x)  # B L C
        x = self.dropout(x)
        x = self.linear(x)
        x = self.act(x)
        return x

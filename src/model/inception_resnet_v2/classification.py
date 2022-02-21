from turtle import forward
from .base_model_3d import InceptionResNetV2 as InceptionResNetV23D
from .layers import TransformerEncoder
from torch import nn
from einops import rearrange


class InceptionResNetV2Transformer3D(nn.Module):
    def __init__(self, n_input_channels, block_size=16,
                 padding='valid', z_channel_preserve=True,
                 dropout_proba=0.3, num_class=2, include_context=False):
        super().__init__()
        self.base_model = InceptionResNetV23D(n_input_channels=n_input_channels, block_size=block_size,
                                              padding=padding, z_channel_preserve=z_channel_preserve,
                                              include_context=include_context)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

        transformer_layer_list = []

        # current feature shape: [1536 2 14 14]
        # current feature shape: [3072 14 14]
        # current feature shape: [768 28 28]
        # current feature shape: [768 784]
        attn_dim_list = [96 for _ in range(6)]
        num_head_list = [8 for _ in range(6)]
        for attn_dim, num_head in zip(attn_dim_list, num_head_list):
            attn_layer = TransformerEncoder(heads=num_head, dim_head=attn_dim,
                                            dropout=dropout_proba)
            transformer_layer_list.append(attn_layer)
        self.transformer_encoder = nn.Sequential(*transformer_layer_list)

        self.final_linear_sequence = nn.Sequential(
            nn.Linear(768, 512),  # 512?
            nn.Dropout(dropout_proba),
            nn.ReLU6(),
            nn.Linear(512, 256),
            nn.ReLU6(),
            nn.Linear(256, num_class),
            nn.Sigmoid()
        )

    def forward(self, x):
        # shape: (B, 3, 32, 512, 512)
        x = self.base_model(x)
        x = rearrange(x, 'b c z h w -> b (c z) h w')
        x = self.pixel_shuffle(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_encoder(x)
        x = x.mean(1)
        output = self.final_linear_sequence(x)

        return output

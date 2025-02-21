import numpy as np
import torch
from torch import nn
from torch.nn import init
from .nn import timestep_embedding
from .diffusion_layer import default, Return
from .diffusion_layer import feature_z_normalize
from src.model.inception_resnet_v2.common_module.layers import get_act, get_norm
from einops import rearrange, repeat

class MLPSkipNet(nn.Module):
    """
    concat x to hidden layers

    default MLP for the latent DPM in the paper!
    """
    def __init__(self, emb_channel, num_time_layers=2, num_time_emb_channels=64, time_last_act=None,
                 activation="silu", use_norm=True, num_hid_channels=1024,
                 num_latent_layers=10, latent_last_act=None, latent_dropout=0., latent_condition_bias=1,
                 skip_layers=[1, 2, 3, 4, 5, 6, 7, 8, 9]):
        super().__init__()
        self.num_time_emb_channels = num_time_emb_channels
        self.skip_layers = skip_layers
        layers = []
        for i in range(num_time_layers):
            if i == 0:
                a = num_time_emb_channels
                b = emb_channel
            else:
                a = emb_channel
                b = emb_channel
            layers.append(nn.Linear(a, b))
            if i < num_time_layers - 1 or (time_last_act is not None):
                layers.append(get_act(time_last_act))
        self.time_embed = nn.Sequential(*layers)

        self.layers = nn.ModuleList([])
        for i in range(num_latent_layers):
            if i == 0:
                act = activation
                norm = use_norm
                cond = True
                a, b = emb_channel, num_hid_channels
                dropout = latent_dropout
            elif i == num_latent_layers - 1:
                act = None
                norm = False
                cond = False
                a, b = num_hid_channels, emb_channel
                dropout = 0
            else:
                act = activation
                norm = use_norm
                cond = True
                a, b = num_hid_channels, num_hid_channels
                dropout = latent_dropout

            if i in skip_layers:
                a += emb_channel

            self.layers.append(
                MLPLNAct(
                    a,
                    b,
                    norm=norm,
                    activation=act,
                    cond_channels=emb_channel,
                    use_cond=cond,
                    condition_bias=latent_condition_bias,
                    dropout=dropout,
                ))
        self.last_act = get_act(latent_last_act)

    def forward(self, x, t, **kwargs):
        t = timestep_embedding(t, self.num_time_emb_channels)
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
    def __init__(self, feature_channel, num_cls=None, z_normalize=True):
        super().__init__()
        self.feature_channel = feature_channel
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
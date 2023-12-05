import math
import torch
from torch import nn
import numpy as np
from ..common_module.layers import get_act
from ..common_module.layers import AttentionPool

USE_INPLACE = True


class ClassificationHeadSimple(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_proba, act):
        super(ClassificationHeadSimple, self).__init__()
        # Global Average Pooling Layer
        self.gap_layer = nn.AdaptiveAvgPool3d((1, 1, 1))

        # First fully connected layer
        self.fc_1 = nn.Linear(in_channels, in_channels * 2)
        self.bn_1 = nn.BatchNorm1d(in_channels * 2)
        self.act_1 = get_act(act)

        # Second fully connected layer
        self.fc_2 = nn.Linear(in_channels * 2, in_channels)
        self.bn_2 = nn.BatchNorm1d(in_channels)
        self.act_2 = nn.ReLU6(inplace=USE_INPLACE)

        # Dropout layer
        self.dropout_layer = nn.Dropout(p=dropout_proba, inplace=USE_INPLACE)

        # Third fully connected layer
        self.fc_3 = nn.Linear(in_channels, in_channels // 2)
        self.bn_3 = nn.BatchNorm1d(in_channels // 2)
        self.act_3 = get_act(act)

        # Output layer
        self.fc_out = nn.Linear(in_channels // 2, num_classes)

    def forward(self, x):
        x = self.gap_layer(x)
        x = x.flatten(start_dim=1, end_dim=-1)

        x = self.fc_1(x)
        x = self.bn_1(x)
        x = self.act_1(x)

        x = self.fc_2(x)
        x = self.bn_2(x)
        x = self.act_2(x)

        x = self.dropout_layer(x)

        x = self.fc_3(x)
        x = self.bn_3(x)
        x = self.act_3(x)

        x = self.fc_out(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, feature_shape, in_channels, num_classes, dropout_proba, act):
        super(ClassificationHead, self).__init__()
        self.attn_pool = AttentionPool(feature_num=np.prod(feature_shape), embed_dim=in_channels,
                                       num_heads=4, output_dim=in_channels * 2)
        self.dropout = nn.Dropout(p=dropout_proba, inplace=USE_INPLACE)
        self.fc = nn.Linear(in_channels * 2, num_classes)
        self.act = get_act(act)

    def forward(self, x):
        x = self.attn_pool(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.act(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=USE_INPLACE)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

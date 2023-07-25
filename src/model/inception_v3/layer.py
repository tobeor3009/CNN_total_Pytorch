import torch
import torch.nn as nn
import torch.nn.functional as F

class ActLayer(nn.Module):
    def __init__(self, activation):
        super(ActLayer, self).__init__()
        if activation is None:
            self.act_layer = nn.Identity()
        elif activation == 'relu':
            self.act_layer = nn.ReLU()
        elif activation == 'leakyrelu':
            self.act_layer = nn.LeakyReLU(0.3)
        else:
            raise ValueError("Unknown activation: {}".format(activation))

    def forward(self, x):
        return self.act_layer(x)


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size, strides=(1, 1),
                 padding='same',  normalization="InstanceNorm2d", 
                 activation='relu', name=None):
        super(ConvBlock2D, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size, 
                              stride=strides, 
                              padding=padding, bias=False)
        
        if normalization == "InstanceNorm2d":
            self.norm = nn.InstanceNorm2d(out_channels)
        elif normalization == "BatchNorm2d":
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            raise ValueError("Unknown normalization: {}".format(normalization))

        self.act = ActLayer(activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

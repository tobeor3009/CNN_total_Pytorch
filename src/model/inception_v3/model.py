import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import ActLayer, ConvBlock2D


class InceptionV3(nn.Module):
    def __init__(self, input_shape=(3, 512, 512),
                 recon_class_num=3, seg_class_num=1, classification_class_num=1,
                 base_act="relu", image_last_act="tanh", last_act="sigmoid",
                 multi_task=False, classification=False):
        super(InceptionV3, self).__init__()

        self.conv_level_1_1 = ConvBlock2D(input_shape[0], 32,
                                          kernel_size=(3, 3), strides=(2, 2),
                                          activation=base_act)
        self.conv_level_2_1 = ConvBlock2D(32, 64,
                                          kernel_size=(3, 3),
                                          activation=base_act)
        self.maxpool_level_2_2 = nn.MaxPool2d(kernel_size=(3, 3),
                                              stride=(2, 2), padding=(1, 1))
        self.conv_level_3_1 = ConvBlock2D(64, 80,
                                          kernel_size=(1, 1),
                                          activation=base_act)
        self.conv_level_3_2 = ConvBlock2D(80, 192,
                                          kernel_size=(3, 3),
                                          activation=base_act)
        self.maxpool_level_3_3 = nn.MaxPool2d(kernel_size=(3, 3),
                                              stride=(2, 2), padding=(1, 1))
        self.branch1x1_level_4_1 = ConvBlock2D(192, 64,
                                               kernel_size=(3, 3),
                                               activation=base_act)
        self.branch5x5_level_4_1 = ConvBlock2D(192, 48,
                                               kernel_size=(1, 1),
                                               activation=base_act)
        self.branch5x5_level_4_2 = ConvBlock2D(48, 64,
                                               kernel_size=(5, 5),
                                               activation=base_act)
        self.branch3x3db1_level_4_1 = ConvBlock2D(192, 64,
                                                  kernel_size=(1, 1),
                                                  activation=base_act)
        self.branch3x3db1_level_4_2 = ConvBlock2D(64, 96,
                                                  kernel_size=(3, 3),
                                                  activation=base_act)
        self.branch3x3db1_level_4_3 = ConvBlock2D(96, 96,
                                                  kernel_size=(3, 3),
                                                  activation=base_act)
        self.branchpool_level_4_1 = nn.AvgPool2d(kernel_size=(3, 3),
                                                 stride=(1, 1), padding=(1, 1))

        self.branch1x1_level_4_1

    def forward(self, x):
        # Define the forward pass
        # x = self.layer1(x) for example
        # ... additional layers
        return x

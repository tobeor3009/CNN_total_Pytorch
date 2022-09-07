import torch.nn as nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x


class VGG(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG, self).__init__()
        # example Input_shape: [B 3 224 224]
        self.step_1 = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64)
        )
        # shape: [B 128 112 112]
        self.step_2 = nn.Sequential(
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 128)
        )
        # shape: [B 256 56 56]
        self.step_3 = nn.Sequential(
            ConvBlock(128, 256, stride=2),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256)
        )
        # shape: [B 512 28 28]
        self.step_4 = nn.Sequential(
            ConvBlock(256, 512, stride=2),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512)
        )
        # shape: [B 512 14 14]
        self.step_5 = nn.Sequential(
            ConvBlock(512, 512, stride=2),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512)

        )
        # shape: [B 512 7 7]
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        # reshape as [B (512* 7* 7)]
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.relu1 = nn.ReLU(True)
        self.drop1 = nn.Dropout()
        # shape: [B 4096]
        self.fc2 = nn.Linear(4096, 1024)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout()
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        # shape: [B 3 224 224]
        x = self.step_1(x)
        # shape: [B 128 112 112]
        x = self.step_2(x)
        # shape: [B 256 56 56]
        x = self.step_3(x)
        # shape: [B 512 28 28]
        x = self.step_4(x)
        # shape: [B 512 14 14]
        x = self.step_5(x)
        # shape: [B 512 7 7]
        x = self.avgpool(x)
        # shape: [B (512* 7* 7)]
        x = x.view(x.size(0), -1)
        # shape: [B 4096]
        x = self.fc1(x)
        x = self.relu1(x)
        # shape: [B 1024]
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        out = self.classifier(x)
        return out

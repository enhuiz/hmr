import torch
import torch.nn as nn

from hmr.networks.cyclegan.discriminator import NLayerD
from torchvision.models import resnet


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ResNetEncoder(resnet.ResNet):
    def __init__(self, opts):
        super().__init__(resnet.BasicBlock, [2, 2, 2, 2])
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                               stride=2, padding=3,
                               bias=False)
        self.fc = nn.Linear(512, opts.output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.transpose(1, 3)
        x = self.fc(x)
        x = x.transpose(1, 3)

        return x

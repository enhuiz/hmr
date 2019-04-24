import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import conv, deconv, resnet18


class NaiveDiscriminator(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """

    def __init__(self, conv_dim):
        super(NaiveDiscriminator, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        self.conv1 = conv(1, conv_dim, 4, 2, 1)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4, 2, 1)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4, 2, 1)
        self.conv4 = conv(conv_dim * 4, 1, 4, 1, 0, batch_norm=False)
        self.fc = nn.Linear(625, 1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))

        out = self.conv4(out).reshape(len(out), -1)
        out = self.fc(out)

        out = torch.sigmoid(out)
        return out


class ResNetDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(dilation=True)
        self.fc = nn.Linear(401408, 1)

    def forward(self, x):
        out = self.resnet(x)
        out = out[-1]
        out = out.reshape([len(out), -1])
        out = self.fc(out[-1])
        return out

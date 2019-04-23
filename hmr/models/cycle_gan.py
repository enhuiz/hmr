import torch
import torch.nn as nn
import torch.nn.functional as F


def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels,
                                     kernel_size, stride, padding, bias=False))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True, init_zero_weights=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(
            out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class ResnetBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResnetBlock, self).__init__()
        self.conv_layer = conv(
            in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out


class CycleGenerator(nn.Module):
    """Defines the architecture of the generator network.
       Note: Both generators G_XtoY and G_YtoX have the same architecture in this assignment.
    """

    def __init__(self, conv_dim, init_zero_weights):
        super(CycleGenerator, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        # 1. Define the encoder part of the generator (that extracts features from the input image)
        self.conv1 = conv(1, conv_dim // 2, 4,
                          init_zero_weights=init_zero_weights)
        self.conv2 = conv(conv_dim // 2, conv_dim, 4,
                          init_zero_weights=init_zero_weights)

        # 2. Define the transformation part of the generator
        self.resnet_block = ResnetBlock(conv_dim)

        # 3. Define the decoder part of the generator (that builds up the output image from features)
        self.deconv1 = deconv(conv_dim, conv_dim // 2, 4)
        self.deconv2 = deconv(conv_dim // 2, 1, 4, batch_norm=False)

    def forward(self, x):
        """Generates an image conditioned on an input image.

            Input
            -----
                x: BS x 3 x 32 x 32

            Output
            ------
                out: BS x 3 x 32 x 32
        """

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))

        out = F.relu(self.resnet_block(out))

        out = F.relu(self.deconv1(out))
        out = torch.tanh(self.deconv2(out))

        return out


class DCDiscriminator(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """

    def __init__(self, conv_dim):
        super(DCDiscriminator, self).__init__()

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


class CycleGAN(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.G_XtoY = CycleGenerator(
            conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
        self.G_YtoX = CycleGenerator(
            conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
        self.D_X = DCDiscriminator(conv_dim=opts.d_conv_dim)
        self.D_Y = DCDiscriminator(conv_dim=opts.d_conv_dim)

    def g_params(self):
        # Get generator parameters
        return list(self.G_XtoY.parameters()) + list(self.G_YtoX.parameters())

    def d_params(self):
        # Get discriminator parameters
        return list(self.D_X.parameters()) + list(self.D_Y.parameters())

    def forward(self, real_X, real_Y, is_g):
        if is_g:
            real_X_score = None
            real_Y_score = None
        else:
            real_X_score = self.D_X(real_X)
            real_Y_score = self.D_Y(real_Y)

        fake_X = self.G_YtoX(real_Y)
        fake_Y = self.G_XtoY(real_X)

        fake_X_score = self.D_X(fake_X)
        fake_Y_score = self.D_Y(fake_Y)

        if is_g:
            rec_X = self.G_YtoX(fake_Y)
            rec_Y = self.G_XtoY(fake_X)
        else:
            rec_X = None
            rec_Y = None

        return {
            'real_X_score': real_X_score,
            'real_Y_score': real_Y_score,
            'fake_X': fake_X,
            'fake_Y': fake_Y,
            'fake_X_score': fake_X_score,
            'fake_Y_score': fake_Y_score,
            'rec_X': rec_X,
            'rec_Y': rec_Y,
        }

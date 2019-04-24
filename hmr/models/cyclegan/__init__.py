import torch
import torch.nn as nn
import torch.nn.functional as F


from .generator import UPerNet
from .discriminator import ResNet


class CycleGAN(nn.Module):
    def __init__(self):
        super().__init__()

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


class UPerNetCycleGAN(CycleGAN):
    def __init__(self, opts):
        super().__init__()
        self.G_XtoY = generator.UPerNet(opts.g_conv_dim)
        self.G_YtoX = generator.UPerNet(opts.g_conv_dim)
        self.D_X = discriminator.ResNet(opts.d_conv_dim)
        self.D_Y = discriminator.ResNet(opts.d_conv_dim)


class UNetCycleGAN(CycleGAN):
    def __init__(self, opts):
        super().__init__()
        self.G_XtoY = generator.UNet(opts.g_conv_dim)
        self.G_YtoX = generator.UNet(opts.g_conv_dim)
        self.D_X = discriminator.ResNet(opts.d_conv_dim)
        self.D_Y = discriminator.ResNet(opts.d_conv_dim)


def get_model(opts):
    model_dict = {
        'UPerNetCycleGAN': UPerNetCycleGAN,
        'UNetCycleGAN': UNetCycleGAN,
    }
    model = None
    if opts.model in model_dict:
        model = model_dict[opts.model](opts)
    return model

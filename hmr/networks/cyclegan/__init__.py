from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .generator import ResNetG
from .discriminator import NLayerD


class CycleGAN(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.G_XtoY = ResNetG(ngf=opts.g_conv_dim)
        self.G_YtoX = ResNetG(ngf=opts.g_conv_dim)
        self.D_X = NLayerD(1, ndf=opts.d_conv_dim)
        self.D_Y = NLayerD(1, ndf=opts.d_conv_dim)

        self.recon_criterion = nn.L1Loss()

    def gan_criterion(self, pred, target):
        target = torch.ones_like(pred).to(pred.device) * target
        return F.mse_loss(pred, target)

    def g_params(self):
        return list(self.G_XtoY.parameters()) + list(self.G_YtoX.parameters())

    def d_params(self):
        return list(self.D_X.parameters()) + list(self.D_Y.parameters())

    def forward_D(self, real_X, real_Y, opts):
        with torch.no_grad():
            fake_X = self.G_YtoX(real_Y)
            fake_Y = self.G_XtoY(real_X)

        real_X_loss = self.gan_criterion(self.D_X(real_X), 1)
        real_Y_loss = self.gan_criterion(self.D_Y(real_Y), 1)
        real_loss = real_X_loss + real_Y_loss

        fake_X_loss = self.gan_criterion(self.D_X(fake_X), 0)
        fake_Y_loss = self.gan_criterion(self.D_Y(fake_Y), 0)
        fake_loss = fake_X_loss + fake_Y_loss

        loss = 0.5 * (real_loss + fake_loss)

        return {
            'real_loss': real_loss,
            'fake_loss': fake_loss,
            'loss': loss,
        }

    def forward_G(self, real_X, real_Y, opts):
        fake_X = self.G_YtoX(real_Y)
        fake_Y = self.G_XtoY(real_X)

        fake_X_loss = self.gan_criterion(self.D_X(fake_X), 1)
        fake_Y_loss = self.gan_criterion(self.D_Y(fake_Y), 1)
        fake_loss = fake_X_loss + fake_Y_loss

        rec_X = self.G_YtoX(fake_Y)
        rec_Y = self.G_XtoY(fake_X)

        cycle_X_loss = self.recon_criterion(rec_X, real_X)
        cycle_Y_loss = self.recon_criterion(rec_Y, real_Y)
        cycle_loss = opts.lambd_x * cycle_X_loss + opts.lambd_y * cycle_Y_loss

        loss = fake_loss + cycle_loss

        return {
            'fake_loss': fake_loss,
            'cycle_loss': cycle_loss,
            'loss': loss,
        }

    def forward(self, real_X, real_Y):
        fake_X = self.G_YtoX(real_Y)
        fake_Y = self.G_XtoY(real_X)
        rec_X = self.G_YtoX(fake_Y)
        rec_Y = self.G_XtoY(fake_X)

        return {
            'real_X': real_X,
            'real_Y': real_Y,
            'fake_X': fake_X,
            'fake_Y': fake_Y,
            'rec_X': rec_X,
            'rec_Y': rec_Y,
        }

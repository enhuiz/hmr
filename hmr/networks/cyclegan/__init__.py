from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .generator import ResNetG
from .discriminator import NLayerD
from hmr.networks.utils import get_class


class CycleGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.score_criterion_raw = nn.MSELoss()
        self.recon_criterion = nn.L1Loss()

    def score_criterion(self, pred, target):
        return self.score_criterion_raw(pred, target.expand_as(pred))

    def g_params(self):
        return list(self.G_XtoY.parameters()) + list(self.G_YtoX.parameters())

    def d_params(self):
        return list(self.D_X.parameters()) + list(self.D_Y.parameters())

    def forward_D(self, real_X, real_Y):
        zeros = torch.zeros(1).to(real_X.device)
        ones = 1 - zeros

        self.G_XtoY.eval()
        self.G_YtoX.eval()
        self.D_X.train()
        self.D_Y.train()

        fake_X = self.G_YtoX(real_Y)
        fake_Y = self.G_XtoY(real_X)

        real_X_loss = self.score_criterion(self.D_X(real_X), ones)
        real_Y_loss = self.score_criterion(self.D_Y(real_Y), ones)
        real_loss = real_X_loss + real_Y_loss

        fake_X_loss = self.score_criterion(self.D_X(fake_X), zeros)
        fake_Y_loss = self.score_criterion(self.D_Y(fake_Y), zeros)
        fake_loss = fake_X_loss + fake_Y_loss

        loss = 0.5 * (real_loss + fake_loss)

        return {
            'real_loss': real_loss,
            'fake_loss': fake_loss,
            'loss': loss,
        }

    def forward_G(self, real_X, real_Y, lambd):
        zeros = torch.zeros(1).to(real_X.device)
        ones = 1 - zeros

        self.G_XtoY.train()
        self.G_YtoX.train()
        self.D_X.eval()
        self.D_Y.eval()

        fake_X = self.G_YtoX(real_Y)
        fake_Y = self.G_XtoY(real_X)

        fake_X_loss = self.score_criterion(self.D_X(fake_X), ones)
        fake_Y_loss = self.score_criterion(self.D_Y(fake_Y), ones)
        fake_loss = fake_X_loss + fake_Y_loss

        rec_X = self.G_YtoX(fake_Y)
        rec_Y = self.G_XtoY(fake_X)

        cycle_X_loss = self.recon_criterion(rec_X, real_X)
        cycle_Y_loss = self.recon_criterion(rec_Y, real_Y)
        cycle_loss = cycle_X_loss + cycle_Y_loss

        loss = fake_loss + lambd * cycle_loss

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


class ResNetCycleGAN(CycleGAN):
    def __init__(self, opts):
        super().__init__()
        self.G_XtoY = ResNetG(ngf=opts.g_conv_dim)
        self.G_YtoX = ResNetG(ngf=opts.g_conv_dim)
        self.D_X = NLayerD(1, ndf=opts.d_conv_dim)
        self.D_Y = NLayerD(1, ndf=opts.d_conv_dim)


def get_model(opts):
    model_list = [ResNetCycleGAN]
    return get_class(model_list, opts.model)(opts)

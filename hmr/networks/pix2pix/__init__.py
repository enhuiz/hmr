from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from hmr.networks.cyclegan.generator import ResNetG
from hmr.networks.cyclegan.discriminator import NLayerD


class Pix2Pix(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.G_XtoY = ResNetG(ngf=opts.g_conv_dim)
        self.G = self.G_XtoY
        self.D = NLayerD(2, ndf=opts.d_conv_dim)
        self.recon_criterion = nn.L1Loss()

    def gan_criterion(self, pred, target):
        target = torch.ones_like(pred).to(pred.device) * target
        return F.mse_loss(pred, target)

    def g_params(self):
        return list(self.G.parameters())

    def d_params(self):
        return list(self.D.parameters())

    def forward_D(self, real_X, real_Y, opts):
        with torch.no_grad():
            fake_Y = self.G(real_X)

        fake_pair = torch.cat((real_X, fake_Y), 1)
        fake_loss = self.gan_criterion(self.D(fake_pair), 0)

        real_pair = torch.cat((real_X, real_Y), 1)
        real_loss = self.gan_criterion(self.D(real_pair), 1)

        loss = (fake_loss + real_loss) * 0.5

        return {
            'real_loss': real_loss,
            'fake_loss': fake_loss,
            'loss': loss,
        }

    def forward_G(self, real_X, real_Y, opts):
        fake_Y = self.G(real_X)

        fake_pair = torch.cat((real_X, fake_Y), 1)
        fake_loss = self.gan_criterion(fake_pair, 1)

        recon_loss = self.recon_criterion(fake_Y, real_Y)

        loss = fake_loss + opts.lambd * recon_loss

        return {
            'fake_loss': fake_loss,
            'recon_loss': recon_loss,
            'loss': loss,
        }

    def forward(self, real_X, real_Y):
        fake_Y = self.G(real_X)
        return {
            'real_X': real_X,
            'real_Y': real_Y,
            'fake_Y': fake_Y,
        }

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .generator import ResnetG, UPerNetG, UNetG
from .discriminator import NLayerD


class CycleGANBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.score_criterion = nn.MSELoss()
        self.recon_criterion = nn.L1Loss()

    def g_params(self):
        return list(self.G_XtoY.parameters()) + list(self.G_YtoX.parameters())

    def d_params(self):
        return list(self.D_X.parameters()) + list(self.D_Y.parameters())


class CycleGAN(CycleGANBase):
    def __init__(self):
        super().__init__()

    def forward(self, real_X, real_Y, is_d):
        zeros = torch.zeros(len(real_X), 1).to(real_X.device)
        ones = 1 - zeros

        out = defaultdict(lambda: None)

        fake_X = self.G_YtoX(real_Y)
        fake_Y = self.G_XtoY(real_X)

        out['fake_X'] = fake_X
        out['fake_Y'] = fake_Y

        fake_X_score = self.D_X(fake_X)
        fake_Y_score = self.D_Y(fake_Y)

        if is_d:
            real_X_score = self.D_X(real_X)
            real_Y_score = self.D_Y(real_Y)

            real_X_loss = self.score_criterion(real_X_score, ones)
            real_Y_loss = self.score_criterion(real_Y_score, ones)
            out['d_real_loss'] = real_X_loss + real_Y_loss

            fake_X_loss = self.score_criterion(fake_X_score, zeros)
            fake_Y_loss = self.score_criterion(fake_Y_score, zeros)
            out['d_fake_loss'] = fake_X_loss + fake_Y_loss
        else:
            rec_X = self.G_YtoX(fake_Y)
            rec_Y = self.G_XtoY(fake_X)
            out['rec_X'] = rec_X
            out['rec_Y'] = rec_Y

            fake_X_loss = self.score_criterion(fake_X_score, ones)
            fake_Y_loss = self.score_criterion(fake_Y_score, ones)
            out['g_fake_loss'] = fake_X_loss + fake_Y_loss

            X_cycle_loss = self.recon_criterion(rec_X, real_X)
            Y_cycle_loss = self.recon_criterion(rec_Y, real_Y)
            out['g_cycle_loss'] = X_cycle_loss + Y_cycle_loss

        return out


class UPerNetCycleGAN(CycleGAN):
    def __init__(self, opts):
        super().__init__()
        self.G_XtoY = UPerNetG(ngf=opts.g_conv_dim)
        self.G_YtoX = UPerNetG(ngf=opts.g_conv_dim)
        self.D_X = NLayerD(1, opts.d_conv_dim)
        self.D_Y = NLayerD(1, opts.d_conv_dim)


class UNetCycleGAN(CycleGAN):
    def __init__(self, opts):
        super().__init__()
        self.G_XtoY = UNetG(ngf=opts.g_conv_dim)
        self.G_YtoX = UNetG(ngf=opts.g_conv_dim)
        self.D_X = NLayerD(1, opts.d_conv_dim)
        self.D_Y = NLayerD(1, opts.d_conv_dim)


class ResNetCycleGAN(CycleGAN):
    def __init__(self, opts):
        super().__init__()
        self.G_XtoY = ResnetG(ngf=opts.g_conv_dim)
        self.G_YtoX = ResnetG(ngf=opts.g_conv_dim)
        self.D_X = NLayerD(1, opts.d_conv_dim)
        self.D_Y = NLayerD(1, opts.d_conv_dim)


class SemiCycleGAN(CycleGANBase):
    def __init__(self):
        super().__init__()

    def forward(self, real_X, real_Y, is_d):
        zeros = torch.zeros(len(real_X), 1).to(real_X.device)
        ones = 1 - zeros

        out = defaultdict(lambda: None)

        fake_X = self.G_YtoX(real_Y)
        fake_Y = self.G_XtoY(real_X)

        out['fake_X'] = fake_X
        out['fake_Y'] = fake_Y

        D_domain = self.D_X
        D_fair = self.D_Y

        fake_X_domain_score = D_domain(fake_X)
        fake_Y_domain_score = D_domain(fake_Y)
        fake_X_fair_score = D_fair(fake_X)
        fake_Y_fair_score = D_fair(fake_Y)

        if is_d:
            real_X_domain_score = D_domain(real_X)
            real_Y_domain_score = D_domain(real_Y)
            real_X_fair_score = D_fair(real_X)
            real_Y_fair_score = D_fair(real_Y)

            real_X_domain_loss = self.score_criterion(
                real_X_domain_score, zeros)
            real_Y_domain_loss = self.score_criterion(
                real_Y_domain_score, ones)

            real_X_fair_loss = self.score_criterion(real_X_fair_score, ones)
            real_Y_fair_loss = self.score_criterion(real_Y_fair_score, ones)

            out['d_real_loss'] = real_X_domain_loss + \
                real_Y_domain_loss + real_X_fair_loss + real_Y_fair_loss

            fake_X_fair_loss = self.score_criterion(fake_X_fair_score, zeros)
            fake_Y_fair_loss = self.score_criterion(fake_Y_fair_score, zeros)

            out['d_fake_loss'] = fake_X_fair_loss + fake_Y_fair_loss
        else:
            rec_X = self.G_YtoX(fake_Y)
            rec_Y = self.G_XtoY(fake_X)
            out['rec_X'] = rec_X
            out['rec_Y'] = rec_Y

            fake_X_domain_loss = self.score_criterion(
                fake_X_domain_score, zeros)
            fake_Y_domain_loss = self.score_criterion(
                fake_Y_domain_score, ones)

            fake_X_fair_loss = self.score_criterion(fake_X_fair_score, ones)
            fake_Y_fair_loss = self.score_criterion(fake_Y_fair_score, ones)

            out['g_fake_loss'] = fake_X_domain_loss + \
                fake_Y_domain_loss + fake_X_fair_loss + fake_Y_fair_loss

            X_cycle_loss = self.recon_criterion(rec_X, real_X)
            Y_cycle_loss = self.recon_criterion(rec_Y, real_Y)
            out['g_cycle_loss'] = X_cycle_loss + Y_cycle_loss

        return out


class ResNetSemiCycleGAN(SemiCycleGAN):
    def __init__(self, opts):
        super().__init__()
        self.G_XtoY = ResnetG(ngf=opts.g_conv_dim)
        self.G_YtoX = ResnetG(ngf=opts.g_conv_dim)
        self.D_X = NLayerD(1, opts.d_conv_dim)
        self.D_Y = NLayerD(1, opts.d_conv_dim)


class UPerNetSemiCycleGAN(SemiCycleGAN):
    def __init__(self, opts):
        super().__init__()
        self.G_XtoY = UPerNetG(ngf=opts.g_conv_dim)
        self.G_YtoX = UPerNetG(ngf=opts.g_conv_dim)
        self.D_X = NLayerD(1, opts.d_conv_dim)
        self.D_Y = NLayerD(1, opts.d_conv_dim)


class UNetSemiCycleGAN(SemiCycleGAN):
    def __init__(self, opts):
        super().__init__()
        self.G_XtoY = UNetG(ngf=opts.g_conv_dim)
        self.G_YtoX = UNetG(ngf=opts.g_conv_dim)
        self.D_X = NLayerD(1, opts.d_conv_dim)
        self.D_Y = NLayerD(1, opts.d_conv_dim)


def get_model(opts):
    model_list = [UPerNetCycleGAN,
                  UNetCycleGAN,
                  ResNetCycleGAN,
                  ResNetSemiCycleGAN,
                  UPerNetSemiCycleGAN,
                  UNetSemiCycleGAN]

    model_dict = {model.__name__.lower(): model for model in model_list}
    model = None
    name = opts.model.lower()
    if name in model_dict:
        model = model_dict[name](opts)
    return model

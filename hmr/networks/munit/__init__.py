"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import os


from .utils import weights_init

##################################################################################
# Discriminator
##################################################################################


class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim, params):
        super(MsImageDis, self).__init__()
        self.n_layer = params.n_layer
        self.dim = params.dim
        self.num_scales = params.num_scales
        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(
            3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0
        for out0, out1 in zip(outs0, outs1):
            loss += F.mse_loss(out0, torch.zeros_like(out0)) + \
                F.mse_loss(out1, torch.ones_like(out1))
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        loss = 0
        for out0 in outs0:
            loss += F.mse_loss(out0, torch.ones_like(out0))
        return loss

##################################################################################
# Generator
##################################################################################


class AdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, params):
        super(AdaINGen, self).__init__()
        dim = params.dim
        style_dim = params.style_dim
        n_downsample = params.n_downsample
        n_res = params.n_res

        # style encoder
        self.enc_style = StyleEncoder(
            4, input_dim, dim, style_dim)

        # content encoder
        self.enc_content = ContentEncoder(
            n_downsample, n_res, input_dim, dim)

        self.dec = Decoder(n_downsample, n_res,
                           self.enc_content.output_dim,
                           input_dim)

    def forward(self, images):
        # reconstruct an image
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode(self, images):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        content = self.enc_content(images)
        return content, style_fake

    def decode(self, content, style):
        # decode content and style codes to an image
        images = self.dec(content)
        return images


##################################################################################
# Encoder and Decoders
##################################################################################

class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1)]
        self.model += [nn.AdaptiveAvgPool2d(1)]  # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

##################################################################################
# Sequential Models
##################################################################################


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


##################################################################################
# Basic Blocks
##################################################################################


class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        self.pad = nn.ZeroPad2d(padding)
        self.activation = nn.ReLU()
        self.conv = nn.Conv2d(input_dim, output_dim,
                              kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.fc(x)
        out = self.activation(out)
        return out


class MUNIT(nn.Module):
    def __init__(self, opts):
        super(MUNIT, self).__init__()

        # Initiate the networks
        # auto-encoder for domain a
        self.gen_a = AdaINGen(1, opts.gen)
        # auto-encoder for domain b
        self.gen_b = AdaINGen(1, opts.gen)
        # discriminator for domain a
        self.dis_a = MsImageDis(1, opts.dis)
        # discriminator for domain b
        self.dis_b = MsImageDis(1, opts.dis)

        self.style_dim = opts.gen.style_dim

        # fix the noise used in sampling
        display_size = int(opts.display_size)
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1)
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1)

        # Network weight initialization
        self.apply(weights_init('gaussian'))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        device = x_a.device
        self.eval()
        s_a1 = self.s_a.to(device)
        s_b1 = self.s_b.to(device)
        s_a2 = torch.randn(x_a.size(0), self.style_dim, 1, 1).to(device)
        s_b2 = torch.randn(x_b.size(0), self.style_dim, 1, 1).to(device)
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
            x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
            x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
            x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))
            x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
            x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()

        return {
            "x_a": x_a,
            "x_a_recon": x_a_recon,
            "x_ab1": x_ab1,
            "x_ab2": x_ab2,
            "x_b": x_b,
            "x_b_recon": x_b_recon,
            "x_ba1": x_ba1,
            "x_ba2": x_ba2
        }

    def forward_G(self, x_a, x_b, opts):
        device = opts.device

        s_a = torch.randn(x_a.size(0), self.style_dim, 1, 1).to(device)
        s_b = torch.randn(x_b.size(0), self.style_dim, 1, 1).to(device)

        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)

        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a, s_a_prime)
        x_b_recon = self.gen_b.decode(c_b, s_b_prime)

        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)

        # encode again
        c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
        c_a_recon, s_b_recon = self.gen_b.encode(x_ab)

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)

        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)

        # total loss
        self.loss = opts.gan_w * self.loss_gen_adv_a + \
            opts.gan_w * self.loss_gen_adv_b + \
            opts.recon_x_w * self.loss_gen_recon_x_a + \
            opts.recon_s_w * self.loss_gen_recon_s_a + \
            opts.recon_c_w * self.loss_gen_recon_c_a + \
            opts.recon_x_w * self.loss_gen_recon_x_b + \
            opts.recon_s_w * self.loss_gen_recon_s_b + \
            opts.recon_c_w * self.loss_gen_recon_c_b

        return {
            'loss': self.loss,
        }

    def forward_D(self, x_a, x_b, opts):
        device = opts.device

        s_a = torch.randn(x_a.size(0), self.style_dim, 1, 1).to(device)
        s_b = torch.randn(x_b.size(0), self.style_dim, 1, 1).to(device)

        # encode
        c_a, _ = self.gen_a.encode(x_a)
        c_b, _ = self.gen_b.encode(x_b)

        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)

        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)

        self.loss = opts.gan_w * (self.loss_dis_a + self.loss_dis_b)

        return {
            'loss': self.loss
        }

    def d_params(self):
        return list(self.dis_a.parameters()) + list(self.dis_b.parameters())

    def g_params(self):
        return list(self.gen_a.parameters()) + list(self.gen_b.parameters())


def get_model(opts):
    model_list = [MUNIT]
    return get_class(model_list, opts.model)(opts)

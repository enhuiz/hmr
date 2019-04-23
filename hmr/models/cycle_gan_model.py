import itertools

from . import networks

import torch
import torch.nn as nn


def get_gpu_ids(device):
    ids = []
    if 'cuda' in device:
        ids.append(int(device.split(':')[1]))
    return ids


class CycleGANModel(nn.Module):
    def __init__(self, input_nc, output_nc, lr, device='cpu',
                 lambda_A=10, lambda_B=10, beta1=0.5,
                 ngf=16, ndf=16, netD='basic', netG='resnet_6blocks',
                 n_layers_D=3, dropout=True, norm='instance',
                 init_type='normal', init_gain=0.2, gan_mode="lsgan"):
        """Initialize the CycleGAN class.
        """
        super().__init__()
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.device = device
        self.gpu_ids = get_gpu_ids(device)

        self.netG_A = networks.define_G(input_nc, output_nc, ngf, netG, norm,
                                        dropout, init_type, init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(output_nc, input_nc, ngf, netG, norm,
                                        dropout, init_type, init_gain, self.gpu_ids)
        self.netD_A = networks.define_D(output_nc, ndf, netD,
                                        n_layers_D, norm, init_type, init_gain, self.gpu_ids)
        self.netD_B = networks.define_D(input_nc, ndf, netD,
                                        n_layers_D, norm, init_type, init_gain, self.gpu_ids)

        # define loss functions
        # define GAN loss.
        self.criterion_gan = networks.GANLoss(gan_mode).to(self.device)
        self.criterion_cycle = torch.nn.L1Loss()

        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(
        ), self.netG_B.parameters()), lr=lr, betas=(beta1, 0.999))

        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(
        ), self.netD_B.parameters()), lr=lr, betas=(beta1, 0.999))

        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)

    def forward_(self):
        """Run forward pass; called by both functions <optimize_parameters>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterion_gan(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterion_gan(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_A = self.lambda_A
        lambda_B = self.lambda_B

        self.loss_G_A = self.criterion_gan(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterion_gan(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterion_cycle(
            self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterion_cycle(
            self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + \
            self.loss_cycle_A + self.loss_cycle_B

        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward_()      # compute fake images and reconstruction images.
        # G_A and G_B
        # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

        return {
            "loss": {
                "g": self.loss_G.cpu().item(),
                "da": self.loss_D_A.cpu().item(),
                "db": self.loss_D_B.cpu().item(),
            },
            "image": {
                "real_A": self.real_A.cpu().detach().numpy(),
                "real_B": self.real_B.cpu().detach().numpy(),
                "fake_A": self.fake_A.cpu().detach().numpy(),
                "fake_B": self.fake_B.cpu().detach().numpy(),
                "rec_A": self.rec_A.cpu().detach().numpy(),
                "rec_B": self.rec_B.cpu().detach().numpy(),
            }
        }

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
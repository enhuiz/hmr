import os
import re
import json
import numpy as np
import argparse
import sys
import glob
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import tqdm
from tensorboardX import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from hmr.data.dataset import MathDataset
from hmr.models.cycle_gan import CycleGAN


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir')
    parser.add_argument('--name', default='cyclegan')
    parser.add_argument('--model', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--mean', type=float, nargs=1, default=[0.5])
    parser.add_argument('--continued', type=bool, default=True)
    parser.add_argument('--sample-every', type=int, default=10)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--init_zero_weights', type=bool, default=True)
    opts = parser.parse_args()
    return opts


def visualize(model, sample, writer, iterations, device):
    real_X = sample['written'].to(device)
    real_Y = sample['printed'].to(device)
    outputs = model(real_X, real_Y, is_g=True)

    def select(x):
        return x[0]

    writer.add_image('real_X', select(real_X), iterations)
    writer.add_image('fake_X', select(outputs['fake_X']), iterations)
    writer.add_image('rec_X', select(outputs['rec_X']), iterations)
    writer.add_image('real_Y', select(real_Y), iterations)
    writer.add_image('fake_Y', select(outputs['fake_Y']), iterations)
    writer.add_image('rec_Y', select(outputs['rec_Y']), iterations)


def train_gan(model, criterion, g_optimizer, d_optimizer, dl, opts):
    writer = SummaryWriter('runs/{}'.format(opts.name))
    sample_to_visual = None
    iterations = 0

    for epoch in range(opts.epoch0, opts.epochs):
        pbar = tqdm.tqdm(dl, total=len(dl))
        for step, sample in enumerate(pbar):
            real_X = sample['written'].to(opts.device)
            real_Y = sample['printed'].to(opts.device)
            ones = torch.ones(opts.batch_size).to(opts.device)
            zeros = torch.zeros(opts.batch_size).to(opts.device)

            # Train D
            outputs = model(real_X, real_Y, is_g=False)

            # Train with real images
            D_X_loss = criterion(outputs['real_X_score'], ones)
            D_Y_loss = criterion(outputs['real_Y_score'], ones)
            d_real_loss = D_X_loss + D_Y_loss
            d_real_loss.backward()

            # Train with fake images
            d_optimizer.zero_grad()
            D_X_loss = criterion(outputs['fake_X_score'], zeros)
            D_Y_loss = criterion(outputs['fake_Y_score'], zeros)
            d_fake_loss = D_X_loss + D_Y_loss
            d_fake_loss.backward()

            d_optimizer.step()

            # Train G
            outputs = model(real_X, real_Y, is_g=True)

            # Train with Y--X-->Y CYCLE
            g_optimizer.zero_grad()
            g_loss = criterion(outputs['fake_X_score'], ones)
            cycle_consistency_loss = criterion(outputs['rec_Y'], real_Y)
            g_loss += cycle_consistency_loss
            g_loss.backward()

            # Train with X--Y-->X CYCLE
            g_optimizer.zero_grad()
            g_loss = criterion(outputs['fake_Y_score'], ones)
            cycle_consistency_loss = criterion(outputs['rec_X'], real_X)
            g_loss += cycle_consistency_loss
            g_loss.backward()

            g_optimizer.step()

            iterations += 1
            pbar.set_description('Epoch [{}/{}], d_real_loss: {:6.4f} | d_Y_loss: {:6.4f} | d_X_loss: {:6.4f} | '
                                 'd_fake_loss: {:6.4f} | g_loss: {:6.4f}'.format(epoch,
                                                                                 opts.epochs,
                                                                                 d_real_loss.item(),
                                                                                 D_Y_loss.item(),
                                                                                 D_X_loss.item(),
                                                                                 d_fake_loss.item(),
                                                                                 g_loss.item()))

            writer.add_scalar('loss/d_real', d_real_loss.item(), iterations)
            writer.add_scalar('loss/D_Y', D_Y_loss.item(), iterations)
            writer.add_scalar('loss/D_X', D_X_loss.item(), iterations)
            writer.add_scalar('loss/d_fake', d_fake_loss.item(), iterations)
            writer.add_scalar('loss/g', g_loss.item(), iterations)

            if step % opts.sample_every == 0:
                if sample_to_visual is None:
                    sample_to_visual = sample
                visualize(model, sample, writer, iterations, opts.device)

        path = os.path.join('checkpoints', opts.name,
                            '{}.pth'.format(epoch + 1))
        torch.save(model, path)


def dump_config(opts):
    config_path = os.path.join(opts.out_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(opts), f)


def get_epoch_num(path):
    epoch = None
    try:
        epoch = int(os.path.basename(path).split('.')[0])
    except:
        pass
    return epoch


def load_model(opts):
    ckpts = glob.glob(os.path.join('checkpoints', opts.name, '*.pth'))
    ckpts = [ckpt for ckpt in ckpts if get_epoch_num(ckpt) is not None]

    if len(ckpts) > 0 and opts.continued:
        ckpt = max(ckpts, key=get_epoch_num)  # latest one
        model = torch.load(ckpt)
        epoch0 = get_epoch_num(ckpt)
        print('Checkpoint {} loaded.'.format(ckpt))
    elif opts.model is None:
        model = CycleGAN(opts)
        epoch0 = 0
        print('Model created.')
    else:
        model = torch.load(opts.model)
        print('Model {} loaded.'.format(opts.model))

    return model, epoch0


def main():
    opts = get_opts()
    os.makedirs('checkpoints', exist_ok=True)
    print(opts)

    model, epoch0 = load_model(opts)
    model = model.to(opts.device)
    opts.epoch0 = epoch0

    g_optimizer = torch.optim.Adam(model.g_params(), opts.lr, [
                                   opts.beta1, opts.beta2])
    d_optimizer = torch.optim.Adam(model.d_params(), opts.lr, [
                                   opts.beta1, opts.beta2])
    criterion = torch.nn.MSELoss()

    dataset = MathDataset(opts.data_dir, 'train')

    dl = DataLoader(dataset, batch_size=opts.batch_size, shuffle=True)

    train_gan(model, criterion, g_optimizer, d_optimizer, dl, opts)


if __name__ == "__main__":
    main()

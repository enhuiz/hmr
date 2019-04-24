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
from torch.utils.data import DataLoader, ConcatDataset, Subset
import matplotlib.pyplot as plt
import tqdm
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from hmr.data.dataset import MathDataset
from hmr.models import cyclegan


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dirs', nargs='+', type=str, default=[])
    parser.add_argument('--name', default='cyclegan')
    parser.add_argument('--model', default='UNetCycleGAN',
                        help='Either a model class name or a ckpt path')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--mean', type=float, nargs=1, default=[0.5])
    parser.add_argument('--continued', type=bool, default=True)
    parser.add_argument('--sample-every', type=int, default=10)
    parser.add_argument('--g_conv_dim', type=int, default=256)
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
        return make_grid(x[:6], 3)

    writer.add_image('1_real_X', select(real_X), iterations)
    writer.add_image('1_fake_Y', select(outputs['fake_Y']), iterations)
    writer.add_image('1_rec_X', select(outputs['rec_X']), iterations)
    writer.add_image('2_real_Y', select(real_Y), iterations)
    writer.add_image('2_fake_X', select(outputs['fake_X']), iterations)
    writer.add_image('2_rec_Y', select(outputs['rec_Y']), iterations)


def train_gan(model, g_optimizer, d_optimizer, dl, opts):
    writer = SummaryWriter('runs/{}'.format(opts.name))
    writer.add_text('opts', json.dumps(vars(opts)))
    sample_to_visual = None
    iterations = 0
    ckpt_dir = os.path.join('checkpoints', opts.name)
    os.makedirs(ckpt_dir, exist_ok=True)

    bcewl_criterion = nn.BCEWithLogitsLoss()
    l1_criterion = nn.L1Loss()

    for epoch in range(opts.epoch0, opts.epochs):
        pbar = tqdm.tqdm(dl, total=len(dl))
        for step, sample in enumerate(pbar):
            real_X = sample['written'].to(opts.device)
            real_Y = sample['printed'].to(opts.device)
            ones = torch.ones(len(real_X), 1).to(opts.device)
            zeros = torch.zeros(len(real_X), 1).to(opts.device)

            # Train D
            d_optimizer.zero_grad()
            outputs = model(real_X, real_Y, is_g=False)

            # Train with real images
            D_X_loss = bcewl_criterion(outputs['real_X_score'], ones)
            D_Y_loss = bcewl_criterion(outputs['real_Y_score'], ones)
            d_real_loss = D_X_loss + D_Y_loss
            d_real_loss.backward()

            # Train with fake images
            D_X_loss = bcewl_criterion(outputs['fake_X_score'], zeros)
            D_Y_loss = bcewl_criterion(outputs['fake_Y_score'], zeros)
            d_fake_loss = D_X_loss + D_Y_loss
            d_fake_loss.backward()

            d_optimizer.step()

            # Train G
            g_optimizer.zero_grad()
            outputs = model(real_X, real_Y, is_g=True)

            G_X_loss = bcewl_criterion(outputs['fake_Y_score'], ones)
            G_Y_loss = bcewl_criterion(outputs['fake_X_score'], ones)
            g_fake_loss = G_X_loss + G_Y_loss

            G_X_cycle_loss = l1_criterion(outputs['rec_X'], real_X)
            G_Y_cycle_loss = l1_criterion(outputs['rec_Y'], real_Y)
            g_cycle_loss = G_X_cycle_loss + G_Y_cycle_loss

            g_loss = g_fake_loss + g_cycle_loss
            g_loss.backward()

            g_optimizer.step()

            iterations += 1
            pbar.set_description('Epoch [{}/{}], '
                                 'd_real_loss: {:6.4f} | '
                                 'd_fake_loss: {:6.4f} | '
                                 'g_fake_loss: {:6.4f} | '
                                 'g_cycle_loss: {:6.4f}'.format(epoch, opts.epochs,
                                                                d_real_loss.item(),
                                                                d_fake_loss.item(),
                                                                g_fake_loss.item(),
                                                                g_cycle_loss.item()))

            writer.add_scalar('loss/d_real', d_real_loss.item(), iterations)
            writer.add_scalar('loss/d_fake', d_fake_loss.item(), iterations)
            writer.add_scalar('loss/g_fake', g_fake_loss.item(), iterations)
            writer.add_scalar('loss/g_cycle', g_cycle_loss.item(), iterations)

            if step % opts.sample_every == 0:
                if sample_to_visual is None:
                    sample_to_visual = sample
                visualize(model, sample_to_visual,
                          writer, iterations, opts.device)

        path = os.path.join(ckpt_dir, '{}.pth'.format(epoch + 1))
        torch.save(model, path)


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
    epoch0 = 0

    if len(ckpts) > 0 and opts.continued:
        ckpt = max(ckpts, key=get_epoch_num)  # latest one
        model = torch.load(ckpt)
        epoch0 = get_epoch_num(ckpt)
        print('Checkpoint {} loaded.'.format(ckpt))
    elif os.path.exists(opts.model):
        model = torch.load(opts.model)
        print('Model {} loaded.'.format(opts.model))
    else:
        model = cyclegan.get_model(opts)
        print('Model created.')

    return model, epoch0


def load_dataset(data_dirs):
    datasets = [MathDataset(d, 'train', paired=False) for d in data_dirs]
    min_len = min(map(len, datasets))
    # balance the dataset
    datasets = [Subset(ds, range(min_len)) for ds in datasets]
    return ConcatDataset(datasets)


def main():
    opts = get_opts()
    print(opts)

    model, epoch0 = load_model(opts)
    model = model.to(opts.device)
    opts.epoch0 = epoch0

    g_optimizer = torch.optim.Adam(model.g_params(), opts.lr, [
                                   opts.beta1, opts.beta2])
    d_optimizer = torch.optim.Adam(model.d_params(), opts.lr, [
                                   opts.beta1, opts.beta2])

    dataset = load_dataset(opts.data_dirs)
    dl = DataLoader(dataset, batch_size=opts.batch_size, shuffle=True)

    train_gan(model, g_optimizer, d_optimizer, dl, opts)


if __name__ == "__main__":
    main()

import os
import re
import json
import numpy as np
import argparse
import sys
import glob
import json
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
import matplotlib.pyplot as plt
import tqdm
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from hmr.data import MathDataset, default_transform
from hmr.networks import munit


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
    parser.add_argument('--g-conv-dim', type=int, default=256)
    parser.add_argument('--d-conv-dim', type=int, default=64)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--init-zero-weights', type=bool, default=True)
    parser.add_argument('--lambd', type=float, default=0.5)
    parser.add_argument('--size', type=int, nargs=2, default=[224, 224])
    opts = parser.parse_args()
    return opts


def visualize(model, sample, writer, iterations, opts):
    real_X = sample['written'].to(opts.device)
    real_Y = sample['printed'].to(opts.device)
    outputs = model(real_X, real_Y, is_d=False)

    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-20)

    def select(x):
        imgs = [normalize(x) for x in x[:6]]
        return make_grid(imgs, math.ceil(len(imgs)**0.5))

    writer.add_image('1_real_X', select(real_X), iterations)
    writer.add_image('1_fake_Y', select(outputs['fake_Y']), iterations)
    writer.add_image('1_rec_X', select(outputs['rec_X']), iterations)
    writer.add_image('2_real_Y', select(real_Y), iterations)
    writer.add_image('2_fake_X', select(outputs['fake_X']), iterations)
    writer.add_image('2_rec_Y', select(outputs['rec_Y']), iterations)


def train_gan(model, g_optimizer, d_optimizer, dl, opts):
    model = model.train()

    writer = SummaryWriter('runs/{}'.format(opts.name))
    writer.add_text('opts', json.dumps(vars(opts)))
    visual_sample = None

    ckpt_dir = os.path.join('ckpt', opts.name)
    os.makedirs(ckpt_dir, exist_ok=True)

    iterations = 0

    for epoch in range(opts.epoch0, opts.epochs):
        pbar = tqdm.tqdm(dl, total=len(dl))
        for step, sample in enumerate(pbar):
            real_X = sample['written'].to(opts.device)
            real_Y = sample['printed'].to(opts.device)

            # Train D
            d_out = model(real_X, real_Y, is_d=True)
            d_optimizer.zero_grad()
            d_out['d_loss'] = d_out['d_real_loss'] + d_out['d_fake_loss']
            d_out['d_loss'].backward()
            d_optimizer.step()

            # Train G
            g_out = model(real_X, real_Y, is_d=False)
            g_optimizer.zero_grad()
            g_out['g_loss'] = g_out['g_fake_loss'] + \
                opts.lambd * g_out['g_cycle_loss']
            g_out['g_loss'].backward()
            g_optimizer.step()

            iterations += 1

            out = list(d_out.items()) + list(g_out.items())
            out_loss = [(k, v) for k, v in out if '_loss' in k]

            texts = []
            for k, v in out_loss:
                writer.add_scalar(k, v.item(), iterations)
                texts.append('{}: {:6.4f}'.format(k, v.item()))

            pbar.set_description('Epoch [{}/{}]'.format(epoch, opts.epochs),
                                 ' | '.join(texts))

            if step % opts.sample_every == 0:
                if visual_sample is None:
                    visual_sample = sample
                visualize(model, visual_sample,
                          writer, iterations, opts)

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
    ckpts = glob.glob(os.path.join('ckpt', opts.name, '*.pth'))
    ckpts = [ckpt for ckpt in ckpts if get_epoch_num(ckpt) is not None]
    epoch0 = 0

    if len(ckpts) > 0 and opts.continued:
        ckpt = max(ckpts, key=get_epoch_num)  # latest one
        model = torch.load(ckpt, map_location='cpu')
        epoch0 = get_epoch_num(ckpt)
        print('Checkpoint {} loaded.'.format(ckpt))
    elif os.path.exists(opts.model):
        model = torch.load(opts.model)
        print('Model {} loaded.'.format(opts.model))
    else:
        model = cyclegan.get_model(opts)
        print('Model created.')

    return model, epoch0


def load_dataset(opts):
    datasets = [MathDataset(d, 'train', default_transform(
        opts.size), paired=False) for d in opts.data_dirs]
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

    g_optimizer = torch.optim.Adam(
        model.g_params(), opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = torch.optim.Adam(
        model.d_params(), opts.lr, [opts.beta1, opts.beta2])

    dataset = load_dataset(opts)
    dl = DataLoader(dataset, batch_size=opts.batch_size, shuffle=True)

    train_gan(model, g_optimizer, d_optimizer, dl, opts)


if __name__ == "__main__":
    main()

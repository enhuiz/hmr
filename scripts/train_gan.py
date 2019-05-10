import os
import re
import numpy as np
import argparse
import sys
import glob
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import transforms
import matplotlib.pyplot as plt
import tqdm

from tensorboardX import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from hmr.data import MathDataset, Vocab
from hmr import networks

from utils import *


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()
    return get_config(args.config)


def train(model, g_optimizer, d_optimizer, wdl, pdl, opts):
    model = model.train()

    writer = SummaryWriter('runs/{}'.format(opts.name))
    writer.add_text('opts', str(opts))
    visual_sample = None

    ckpt_dir = os.path.join('ckpt', opts.name)
    os.makedirs(ckpt_dir, exist_ok=True)

    iterations = opts.epoch0 * len(wdl)
    total_iterations = opts.epochs * len(wdl)

    for epoch in range(opts.epoch0, opts.epochs):
        pbar = tqdm.tqdm(zip(wdl, pdl), total=len(wdl))
        for step, (ws, ps) in enumerate(pbar):
            d_lr = adjust_lr(d_optimizer, iterations, total_iterations, opts)
            g_lr = adjust_lr(g_optimizer, iterations, total_iterations, opts)

            real_X = ws['image'].to(opts.device)
            real_Y = ps['image'].to(opts.device)

            out = {}
            # Train D
            d_optimizer.zero_grad()
            out['d'] = model.forward_D(real_X, real_Y, opts)
            out['d']['loss'].backward()
            d_optimizer.step()

            # Train G
            g_optimizer.zero_grad()
            out['g'] = model.forward_G(real_X, real_Y, opts)
            out['g']['loss'].backward()
            g_optimizer.step()

            iterations += 1

            description = []
            for k, v in flatten_dict(out).items():
                v = v.item()
                writer.add_scalar(k, v, iterations)
                description.append('{}: {:.4f}'.format(k, v))

            description = 'Epoch [{}/{}] | d.lr {:.4f} | g.lr {:.4f} | '.format(
                epoch + 1, opts.epochs, d_lr, g_lr) + ' | '.join(description)
            pbar.set_description(description)

            if step % opts.sample_every == 0:
                if visual_sample is None:
                    visual_sample = (ws, ps)
                visualize(model, *visual_sample,
                          writer, iterations, opts)

        path = os.path.join(ckpt_dir, '{}.pth'.format(epoch + 1))
        torch.save(model, path)


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
        model = networks.get_model(opts)
        print('Model created.')

    return model, epoch0


def create_transform(opts):
    def identity(x): return x

    return transforms.Compose([
        transforms.Resize(opts.base_size),
        identity if opts.paired else transforms.RandomCrop(opts.crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [1]),
    ])


def load_dataloaders(opts):
    wds = MathDataset(opts.data_dir, 'written', 'train',
                      create_transform(opts), opts.vocab)

    pds = MathDataset(opts.data_dir, 'printed', 'train',
                      create_transform(opts), opts.vocab)

    wdl = DataLoader(wds,
                     batch_size=opts.batch_size,
                     shuffle=not opts.paired,
                     collate_fn=wds.get_collate_fn())

    pdl = DataLoader(pds,
                     batch_size=opts.batch_size,
                     shuffle=not opts.paired,
                     collate_fn=pds.get_collate_fn())

    return wdl, pdl


def main():
    opts = get_opts()
    print(opts)
    opts.vocab = Vocab(os.path.join(opts.data_dir, 'annotations', 'vocab.csv'))

    model, epoch0 = load_model(opts)
    model = model.to(opts.device)
    opts.epoch0 = epoch0

    g_optimizer = torch.optim.Adam(
        model.g_params(), opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = torch.optim.Adam(
        model.d_params(), opts.lr, [opts.beta1, opts.beta2])

    wdl, pdl = load_dataloaders(opts)
    train(model, g_optimizer, d_optimizer, wdl, pdl, opts)


if __name__ == "__main__":
    main()

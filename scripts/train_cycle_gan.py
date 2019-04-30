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
from pandas.io.json import json_normalize

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from hmr.data import MathDataset, default_transform
from hmr.networks import cyclegan


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str)
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


def visualize(model, ws, ps, writer, iterations, opts):
    real_X = ws['image'].to(opts.device)
    real_Y = ps['image'].to(opts.device)
    model.eval()
    out = model(real_X, real_Y)
    model.train()

    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-20)

    def select(x):
        imgs = [normalize(x) for x in x[:6]]
        return make_grid(imgs, math.ceil(len(imgs)**0.5))

    for k, v in out.items():
        writer.add_image(k, select(v), iterations)


def adjust_lr(optimizer, interations, total_iterations, opts):
    lr = opts.lr * (1 - interations / total_iterations) ** 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def flatten_dict(d):
    ret = {}
    for k in d:
        for l in d[k]:
            ret['{}.{}'.format(k, l)] = d[k][l]
    return ret


def train(model, g_optimizer, d_optimizer, pdl, wdl, opts):
    model = model.train()

    writer = SummaryWriter('runs/{}'.format(opts.name))
    writer.add_text('opts', json.dumps(vars(opts)))
    visual_sample = None

    ckpt_dir = os.path.join('ckpt', opts.name)
    os.makedirs(ckpt_dir, exist_ok=True)

    iterations = opts.epoch0 * len(wdl)
    total_iterations = opts.epochs * len(wdl)

    for epoch in range(opts.epoch0, opts.epochs):
        pbar = tqdm.tqdm(zip(pdl, wdl), total=len(wdl))
        for step, (ps, ws) in enumerate(pbar):
            real_X = ps['image'].to(opts.device)
            real_Y = ws['image'].to(opts.device)

            out = {}
            # Train D
            d_optimizer.zero_grad()
            out['d'] = model.forward_D(real_X, real_Y)
            out['d']['loss'].backward()
            d_optimizer.step()

            # Train G
            g_optimizer.zero_grad()
            out['g'] = model.forward_G(real_X, real_Y, opts.lambd)
            out['g']['loss'].backward()
            g_optimizer.step()

            iterations += 1

            d_lr = adjust_lr(d_optimizer, iterations, total_iterations, opts)
            g_lr = adjust_lr(g_optimizer, iterations, total_iterations, opts)

            description = []
            for k, v in flatten_dict(out).items():
                v = v.item()
                writer.add_scalar(k, v, iterations)
                description.append('{}: {:6.4f}'.format(k, v))

            description = 'Epoch [{}/{}] | dlr {:6.4f} | glr {:6.4f}'.format(
                epoch + 1, opts.epochs, d_lr, g_lr) + ' | '.join(description)
            pbar.set_description(description)

            if step % opts.sample_every == 0:
                if visual_sample is None:
                    visual_sample = (ps, ws)
                visualize(model, *visual_sample,
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


def load_dataloaders(opts):
    pds = MathDataset(opts.data_dir, 'train',
                      default_transform(opts.size),
                      written=False)
    wds = MathDataset(opts.data_dir, 'train',
                      default_transform(opts.size),
                      written=True)

    assert len(pds) == len(wds)

    pdl = DataLoader(pds,
                     batch_size=opts.batch_size,
                     shuffle=False)

    wdl = DataLoader(wds,
                     batch_size=opts.batch_size,
                     shuffle=True)

    return pdl, wdl


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

    pdl, wdl = load_dataloaders(opts)
    train(model, g_optimizer, d_optimizer, pdl, wdl, opts)


if __name__ == "__main__":
    main()

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
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from hmr.data import MathDataset, Vocab
from hmr.networks import cyclegan

from utils import get_config


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()
    return get_config(args.config)


def flatten_dict(d):
    ret = {}
    for k in d:
        for l in d[k]:
            ret['{}.{}'.format(k, l)] = d[k][l]
    return ret


def forward_once(sample, model, dir_, opts):
    id_ = sample['id']
    real = sample['image'].to(opts.device)
    fake = model(real).detach().cpu().numpy()
    for id_, img in zip(sample['id'], fake):
        path = os.path.join(dir_, '{}.png'.format(id_))
        img = (img + opts.mean).squeeze() * 255
        img = Image.fromarray(img)
        img = img.convert('L')
        img.save(path)


def forward(model, wdl, pdl, opts):
    model = model.eval()
    pbar = tqdm.tqdm(zip(wdl, pdl), total=len(wdl))

    fpdir = os.path.join(opts.out_dir, 'fake_printed', opts.part)
    fwdir = os.path.join(opts.out_dir, 'fake_written', opts.part)
    os.makedirs(fpdir, exist_ok=True)
    os.makedirs(fwdir, exist_ok=True)

    for ws, ps in pbar:
        with torch.no_grad():
            forward_once(ws, model.G_XtoY, fpdir, opts)
            forward_once(ps, model.G_YtoX, fwdir, opts)


def load_model(opts):
    model = torch.load(opts.model, map_location=lambda s, _: s)
    print('Model {} loaded.'.format(opts.model))
    return model


def create_transform(opts):
    return transforms.Compose([
        transforms.Resize(opts.base_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [1]),
    ])


def main():
    opts = get_opts()
    print(opts)
    opts.vocab = Vocab(os.path.join(opts.data_dir, 'annotations', 'vocab.csv'))

    model = load_model(opts)
    model = model.to(opts.device)

    for part in opts.parts:
        opts.part = part

        wds = MathDataset(opts.data_dir, 'written', opts.part,
                          create_transform(opts), opts.vocab)

        wdl = DataLoader(wds,
                         batch_size=opts.batch_size,
                         collate_fn=wds.get_collate_fn(),
                         shuffle=False)

        pds = MathDataset(opts.data_dir, 'printed', opts.part,
                          create_transform(opts), opts.vocab)

        pdl = DataLoader(pds,
                         batch_size=opts.batch_size,
                         collate_fn=pds.get_collate_fn(),
                         shuffle=False)

        forward(model, wdl, pdl, opts)


if __name__ == "__main__":
    main()

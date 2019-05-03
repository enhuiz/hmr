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
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from hmr.data import MathDataset, default_transform
from hmr.networks import cyclegan


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--out-dir', type=str)
    parser.add_argument('--type', type=str, default='dev')
    parser.add_argument('--model')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--mean', type=float, nargs=1, default=[0.5])
    parser.add_argument('--size', type=int, nargs=2, default=[224, 224])
    opts = parser.parse_args()
    return opts


def flatten_dict(d):
    ret = {}
    for k in d:
        for l in d[k]:
            ret['{}.{}'.format(k, l)] = d[k][l]
    return ret


def train(model, dl, opts):
    model = model.train()

    pbar = tqdm.tqdm(dl, total=len(dl))
    for step, sample in enumerate(pbar):
        id_ = sample['id']
        real_X = sample['image'].to(opts.device)
        fake_Y = model.G_XtoY(real_X).detach().cpu().numpy()
        for id_, img in zip(sample['id'], fake_Y):
            path = os.path.join(opts.out_dir, '{}.png'.format(id_))
            img = (img + opts.mean).squeeze() * 255
            img = Image.fromarray(img)
            img = img.convert('L')
            img.save(path)


def load_model(opts):
    model = torch.load(opts.model, map_location=lambda s, _: s)
    print('Model {} loaded.'.format(opts.model))
    return model


def main():
    opts = get_opts()
    os.makedirs(opts.out_dir, exist_ok=True)
    print(opts)

    model = load_model(opts)
    model = model.to(opts.device)

    ds = MathDataset(opts.data_dir, opts.type,
                     default_transform(opts.size),
                     written=True)

    dl = DataLoader(ds,
                    batch_size=opts.batch_size,
                    collate_fn=MathDataset.collate_fn,
                    shuffle=False)

    train(model, dl, opts)


if __name__ == "__main__":
    main()

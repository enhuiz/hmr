import os
import re
import json
import numpy as np
import argparse
import sys
import glob
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from hmr.data.dataset import MathDataset
from hmr.models.cycle_gan_model import CycleGANModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir')
    parser.add_argument('--out-dir')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--mean', type=float, nargs=1, default=[0.5])
    parser.add_argument('--input-type', choices=['printed', 'written'])
    parser.add_argument('--output-type', choices=['printed', 'written'])
    parser.add_argument('--continued', type=bool, default=True)
    args = parser.parse_args()
    return args


def train(model, dl, args):
    for epoch in range(args.epoch0, args.epochs):
        pbar = tqdm.tqdm(dl, total=len(dl))
        for step, sample in enumerate(pbar):
            model.set_input({
                "A": sample['written'],
                "B": sample['printed'],
            })
            
            output = model.optimize_parameters()

            losses = {k: np.round(v, 4) for k, v in output['loss'].items()}
            pbar.set_description('Epoch [{}/{}], '
                                 'Loss: {}'.format(epoch + 1,
                                                   args.epochs,
                                                   losses))
            if step % (500 / args.batch_size) == 0:
                for k, v in output['image'].items():
                    plt.imsave('{}.png'.format(k), v[0][0], cmap='gray')

        path = os.path.join(args.out_dir, '{}.pth'.format(epoch + 1))
        torch.save(model, path)


def dump_config(args):
    config_path = os.path.join(args.out_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f)


def get_epoch_num(path):
    epoch = None
    try:
        epoch = int(os.path.basename(path).split('.')[0])
    except:
        pass
    return epoch


def load_model(args):
    ckpts = glob.glob(os.path.join(args.out_dir, '*.pth'))
    ckpts = [ckpt for ckpt in ckpts if get_epoch_num(ckpt) is not None]
    if len(ckpts) > 0 and args.continued:
        ckpt = max(ckpts, key=get_epoch_num)  # latest one
        model = torch.load(ckpt)
        epoch0 = get_epoch_num(ckpt)
        print('Checkpoint {} loaded. ...'.format(ckpt))
    else:
        model = CycleGANModel(1, 1, args.lr, device=args.device)
        epoch0 = 0
        print('Model created.')
    return model, epoch0


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dump_config(args)
    print(args)

    model, epoch0 = load_model(args)
    args.epoch0 = epoch0

    dataset = MathDataset(args.data_dir, 'train')
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    train(model, dl, args)


if __name__ == "__main__":
    main()

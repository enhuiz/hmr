import os
import torch
import math
import glob
import yaml
from collections import namedtuple
from torchvision.utils import make_grid
import argparse


def adjust_lr(optimizer, interations, total_iterations, opts):
    lr = opts.lr * (1 - interations / total_iterations) ** 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_epoch_num(path):
    epoch = None
    try:
        epoch = int(os.path.basename(path).split('.')[0])
    except:
        pass
    return epoch


def flatten_dict(d):
    ret = {}
    for k in d:
        for l in d[k]:
            ret['{}.{}'.format(k, l)] = d[k][l]
    return ret


def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-20)


def visualize(model, ws, ps, writer, iterations, opts):
    real_X = ws['image'][:opts.display_size].to(opts.device)
    real_Y = ps['image'][:opts.display_size].to(opts.device)
    model.eval()
    out = model(real_X, real_Y)
    model.train()

    def select(x):
        imgs = [normalize(x) for x in x]
        return make_grid(imgs, math.ceil(len(imgs)**0.5))

    for k, v in out.items():
        writer.add_image(k, select(v), iterations)


def make_namespace(d):
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = make_namespace(d[key])
    return argparse.Namespace(**d)


def get_config(config):
    with open(config, 'r') as stream:
        d = yaml.load(stream, Loader=yaml.FullLoader)
    return make_namespace(d)

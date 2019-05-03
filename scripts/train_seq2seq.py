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
import matplotlib.pyplot as plt
import tqdm
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from nltk.metrics import edit_distance
from nltk.translate.bleu_score import sentence_bleu
from torchvision import transforms

torch.backends.cudnn.benchmark = True

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from hmr import vocab
from hmr import networks
from hmr.data import MathDataset

from utils import get_config, get_epoch_num


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()
    return get_config(args.config)


def visualize(model, sample, writer, iterations, opts):
    images = sample['image'][:1].to(opts.device)
    len_ = sample['len'][0]
    captions = sample['caption'][:len_, :1].to(opts.device)

    model.eval()
    out = model.decode(images)
    model.train()

    logp = out['logp']
    predictions = torch.argmax(logp, dim=2)

    print(captions.shape, predictions.shape)
    predictions = predictions.squeeze(1).tolist()
    captions = captions.squeeze(1).tolist()

    ref = list(map(vocab.index2word, captions))
    hyp = list(map(vocab.index2word, predictions))

    bleu = sentence_bleu(ref, hyp)
    ed = edit_distance(ref, hyp)

    print('BLEU: {}, ED: {}\nRef: {}\nHyp: {}'.format(
        bleu, ed, ' '.join(ref), ' '.join(hyp)))

    writer.add_text('sentence', ' '.join(ref) +
                    '\n' + ' '.join(hyp),
                    iterations)
    writer.add_image('image', images[0], iterations)

    weights = out['weights'][:, 0]
    for t in range(len(weights)):
        writer.add_image('weight-{}'.format(iterations), weights[t:t + 1], t)


def adjust_lr(optimizer, interations, total_iterations, opts):
    lr = opts.lr * (1 - interations / total_iterations) ** 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(model, optimizer, dl, opts):
    model = model.train()

    writer = SummaryWriter('runs/{}'.format(opts.name))
    writer.add_text('opts', str(opts))
    visual_sample = None

    ckpt_dir = os.path.join('ckpt', opts.name)
    os.makedirs(ckpt_dir, exist_ok=True)

    iterations = opts.epoch0 * len(dl)
    total_iterations = opts.epochs * len(dl)

    for epoch in range(opts.epoch0, opts.epochs):
        pbar = tqdm.tqdm(dl, total=len(dl))
        for step, sample in enumerate(pbar):
            images = sample['image'].to(opts.device)
            captions = sample['caption'].to(opts.device)
            lens = sample['len'].to(opts.device)

            optimizer.zero_grad()
            out = model(images, captions, lens)
            out['loss'].backward()
            optimizer.step()

            iterations += 1

            lr = adjust_lr(optimizer, iterations, total_iterations, opts)

            description = 'Epoch [{}/{}], lr {:6.4f}, loss: {:6.4f}'.format(
                epoch + 1, opts.epochs, lr, out['loss'])
            pbar.set_description(description)

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
        model = networks.get_model(opts)
        print('Model created.')

    return model, epoch0


def create_transform(opts):
    return transforms.Compose([
        transforms.Resize(opts.base_size),
        transforms.ColorJitter(0.5, 0.5, 0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [1]),
    ])


def load_dataloader(opts):
    ds = MathDataset(opts.data_dir, 'train',
                     create_transform(opts),
                     written=False)

    dl = DataLoader(ds, batch_size=opts.batch_size,
                    shuffle=True, collate_fn=MathDataset.collate_fn)

    return dl


def main():
    opts = get_opts()
    print(opts)

    dl = load_dataloader(opts)

    model, epoch0 = load_model(opts)
    model = model.to(opts.device)
    opts.epoch0 = epoch0
    optimizer = torch.optim.Adam(model.parameters(), opts.lr, [
                                 opts.beta1, opts.beta2])

    train(model, optimizer, dl, opts)


if __name__ == "__main__":
    main()

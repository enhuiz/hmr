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
from tensorboardX import SummaryWriter
from torchvision import transforms
from torchsummary import summary
import tqdm
import matplotlib.pyplot as plt


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from hmr import networks
from hmr.data import Vocab, MathDataset
from utils import get_config, get_epoch_num, denormalize, add_mask, calculate_scores, adjust_lr


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    opts = parser.parse_args()
    return get_config(opts.config)


def visualize(model, sample, writer, iterations, opts):
    images = sample['image'][:1].to(opts.device)
    len_ = sample['len'][0]
    annotations = sample['annotation'][:len_, :1].to(opts.device)

    model.eval()
    with torch.no_grad():
        out = model.decode(images,
                           opts.vocab.word2index('</s>'),
                           opts.max_output_len)
    model.train()

    outputs = out['outputs'][0]
    predictions = torch.argmax(outputs, dim=2)

    prediction = predictions.squeeze(1).tolist()
    annotation = annotations.squeeze(1).tolist()

    hyp = opts.vocab.decode(prediction)
    ref = opts.vocab.decode(annotation)

    content = 'Ref: {}; Hyp: {}'.format(' '.join(ref), ' '.join(hyp))
    writer.add_text('sentence', content, iterations)
    print(calculate_scores([ref], [hyp]))
    print(content)

    image = denormalize(images[0]).cpu()
    writer.add_image('image', image, iterations)

    weights = out['weights'][0][:, 0]
    for t, weight in enumerate(weights):
        weight = weight.unsqueeze(0).cpu()
        masked = add_mask(image, weight)
        writer.add_image('weights-{}'.format(t), masked, iterations)


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
            annotations = sample['annotation'].to(opts.device)
            lens = sample['len'].to(opts.device)

            optimizer.zero_grad()
            out = model(images, annotations, lens)
            out['loss'].backward()
            optimizer.step()

            iterations += 1

            lr = adjust_lr(optimizer, iterations, total_iterations, opts)

            description = 'Epoch [{}/{}], lr {:6.4f}, loss: {:6.4f}'.format(
                epoch + 1, opts.epochs, lr, out['loss'])
            pbar.set_description(description)

            writer.add_scalar('loss', out['loss'].item(), iterations)
            if step % opts.sample_every == 0:
                if visual_sample is None:
                    visual_sample = sample
                visualize(model, visual_sample,
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
    return transforms.Compose([
        transforms.Resize(opts.base_size),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [1]),
    ])


def load_dataloader(opts):
    # only train on printed-train data
    ds = MathDataset(opts.data_dir, 'printed', 'train',
                     create_transform(opts), opts.vocab)

    dl = DataLoader(ds, batch_size=opts.batch_size,
                    shuffle=True, collate_fn=ds.get_collate_fn())

    return dl


def main():
    opts = get_opts()
    print(opts)

    # the vocab of training should be consistent with the training data
    opts.vocab = Vocab(os.path.join(opts.data_dir, 'annotations', 'vocab.csv'))

    dl = load_dataloader(opts)

    model, epoch0 = load_model(opts)
    opts.epoch0 = epoch0

    if hasattr(opts, 'fine_tune') and opts.fine_tune and epoch0 == 0:
        model.decoder.update_output_dim(len(opts.vocab))
        optimizer = torch.optim.Adam(model.decoder.get_fine_tune_params(), opts.lr, [
            opts.beta1, opts.beta2])
    else:
        optimizer = torch.optim.Adam(model.parameters(), opts.lr, [
            opts.beta1, opts.beta2])

    model = model.to(opts.device)
    train(model, optimizer, dl, opts)


if __name__ == "__main__":
    main()

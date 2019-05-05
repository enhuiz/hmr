import os
import re
import numpy as np
import argparse
import sys
import glob
import time
import math

import tqdm
import nltk
import torch
import distance

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from hmr import networks
from hmr.data import vocab, MathDataset
from utils import get_config, get_epoch_num, normalize, add_mask


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()
    return get_config(args.config)


def exact_match_score(references, hypotheses):
    """Computes exact match scores.
    Args:
        references: list of list of tokens (one ref)
        hypotheses: list of list of tokens (one hypothesis)
    Returns:
        exact_match: (float) 1 is perfect
    """
    exact_match = 0
    for ref, hypo in zip(references, hypotheses):
        if np.array_equal(ref, hypo):
            exact_match += 1

    return exact_match / float(max(len(hypotheses), 1))


def bleu_score(references, hypotheses):
    """Computes bleu score.
    Args:
        references: list of list (one hypothesis)
        hypotheses: list of list (one hypothesis)
    Returns:
        BLEU-4 score: (float)
    """
    references = [[ref] for ref in references]  # for corpus_bleu func
    BLEU_4 = nltk.translate.bleu_score.corpus_bleu(references, hypotheses,
                                                   weights=(0.25, 0.25, 0.25, 0.25))
    return BLEU_4


def edit_similarity(references, hypotheses):
    """Computes Levenshtein distance between two sequences.
    Args:
        references: list of list of token (one hypothesis)
        hypotheses: list of list of token (one hypothesis)
    Returns:
        1 - levenshtein distance: (higher is better, 1 is perfect)
    """
    d_leven, len_tot = 0, 0
    for ref, hypo in zip(references, hypotheses):
        d_leven += distance.levenshtein(ref, hypo)
        len_tot += float(max(len(ref), len(hypo)))

    return 1. - d_leven / len_tot


def write_sentences(path, sentences):
    with open(path, 'w') as f:
        f.write('\n'.join(sentences))


def evaluate(model, dl, opts):
    model = model.eval()
    hyps = []
    refs = []

    for sample in tqdm.tqdm(dl, total=len(dl)):
        images = sample['image'].to(opts.device)
        annotations = sample['annotation'].to(opts.device)
        lens = sample['len'].to(opts.device)

        with torch.no_grad():
            out = model.decode(images)

        outputs = out['outputs']

        for i in range(len(images)):
            prediction = torch.argmax(outputs[i], dim=-1).squeeze()
            annotation = annotations[:lens[i], i]
            hyps.append(prediction.tolist())
            refs.append(annotation.tolist())

    scores = {
        "BLEU-4": bleu_score(refs, hyps) * 100,
        "EM": exact_match_score(refs, hyps) * 100,
        "Edit": edit_similarity(refs, hyps) * 100
    }

    refs = [' '.join(map(vocab.index2word, ref)) for ref in refs]
    hyps = [' '.join(map(vocab.index2word, hyp)) for hyp in hyps]

    results_dir = 'results/{}'.format(opts.name)
    os.makedirs(results_dir, exist_ok=True)

    write_sentences(os.path.join(results_dir, 'refs.txt'), refs)
    write_sentences(os.path.join(results_dir, 'hyps.txt'), hyps)
    with open(os.path.join(results_dir, 'scores.txt'), 'w') as f:
        f.write(str(scores))

    print(scores)


def create_transform(opts):
    return transforms.Compose([
        transforms.Resize(opts.base_size),
        transforms.ToTensor(),
        transforms.Normalize(opts.mean, [1]),
    ])


def load_dataloader(opts):
    ds = MathDataset(opts.data_dir, 'dev',
                     create_transform(opts),
                     written=False)

    dl = DataLoader(ds, batch_size=opts.batch_size,
                    shuffle=True, collate_fn=MathDataset.collate_fn)

    return dl


def extract_epoch_num(path):
    return int(path.split('/')[-1].rstrip('.pth'))


def main():
    opts = get_opts()
    print(opts)

    dl = load_dataloader(opts)

    ckpts = glob.glob(os.path.join('ckpt', opts.name, '*.pth'))
    ckpt = sorted(ckpts, key=extract_epoch_num)[-1]
    model = torch.load(ckpt, map_location='cpu')
    model = model.to(opts.device)

    print('{} loaded.'.format(ckpt))

    evaluate(model, dl, opts)


if __name__ == "__main__":
    main()

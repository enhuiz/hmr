import os
import torch
import math
import glob
import yaml
from collections import namedtuple
import argparse

import nltk
import distance
import numpy as np
import torch.nn.functional as F
from torchvision.utils import make_grid


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


def normalize(x, opts):
    return x + torch.tensor(opts.mean).to(x.device)


def visualize(model, ws, ps, writer, iterations, opts):
    real_X = ws['image'][:opts.display_size].to(opts.device)
    real_Y = ps['image'][:opts.display_size].to(opts.device)
    model.eval()
    out = model(real_X, real_Y)
    model.train()

    def select(x):
        imgs = [normalize(x, opts) for x in x]
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


def add_mask(image, weight):
    mask = F.interpolate(weight.unsqueeze(0),
                         image.shape[1:],
                         mode='bilinear', align_corners=True).squeeze()
    masked = image * (mask * 0.8 + 0.2)
    return masked


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


def calculate_scores(refs, hyps):
    scores = {
        "BLEU-4": bleu_score(refs, hyps) * 100,
        "EM": exact_match_score(refs, hyps) * 100,
        "Edit": edit_similarity(refs, hyps) * 100
    }
    return scores

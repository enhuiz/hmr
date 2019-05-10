import os
import re
import numpy as np
import pandas as pd

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence


class Vocab():
    def __init__(self, path):
        self.load(path)

    def load(self, path):
        with open(path, 'r') as f:
            content = f.read().strip()
        words = content.split('\n')
        words = set(words)

        extra = set()
        extra.add('<s>')
        extra.add('</s>')
        extra.add('<pad>')
        extra.add('<unk>')

        self.extra = extra
        self.words = sorted(words.union(extra))
        self.inv_words = {w: i for i, w in enumerate(self.words)}

    def word2index(self, w):
        if w not in self.inv_words:
            w = '<unk>'
        return self.inv_words[w]

    def index2word(self, i):
        w = self.words[i]
        return w

    def not_extra(self, w):
        return w not in self.extra

    def decode(self, ids):
        return [*filter(self.not_extra, map(self.index2word, ids))]

    def __len__(self):
        return len(self.words)

    def __str__(self):
        return str(self.words)


def create_samples(data_dir, style, part):
    path = os.path.join(data_dir, 'annotations', '{}.csv'.format(part))
    df = pd.read_csv(path, names=['id', 'annotation'])

    image_path = os.path.join(data_dir, 'features', style, part, '{}.png')
    df['image'] = df['id'].apply(image_path.format)

    assert all(df['image'].apply(os.path.exists))  # assert existence

    df['annotation'] = df['annotation'].str.strip()
    samples = df[['id', 'image', 'annotation']].to_dict('record')

    return samples


class MathDataset(Dataset):
    def __init__(self, data_dir, style, part, transform, vocab):
        self.transform = transform
        self.vocab = vocab
        self.samples = create_samples(data_dir, style, part)

    @staticmethod
    def load_pil(path):
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('L')
        except Exception as e:
            # some image in im2latex maybe broken, so create an empty image here
            print(e)
            return Image.new('L', (1, 1))

    def process_annotation(self, annotation):
        annotation = '<s> {} </s>'.format(annotation).split(' ')
        annotation = list(map(self.vocab.word2index, annotation))
        annotation = torch.tensor(annotation)
        return annotation

    def __getitem__(self, idx):
        sample = dict(self.samples[idx])
        image = self.transform(self.load_pil(sample['image']))
        annotation = self.process_annotation(sample['annotation'])
        sample['image'] = image
        sample['annotation'] = annotation
        sample['len'] = len(annotation)
        return sample

    def __len__(self):
        return len(self.samples)

    def get_collate_fn(self):
        pad_ix = self.vocab.word2index('<pad>')

        def collate_fn(batch):
            image = [sample['image'] for sample in batch]
            annotation = [sample['annotation'] for sample in batch]
            len_ = [sample['len'] for sample in batch]
            id_ = [sample['id'] for sample in batch]

            ret = {}

            ret['image'] = torch.stack(image)
            ret['annotation'] = pad_sequence(annotation, False, pad_ix)
            ret['len'] = torch.tensor(len_)
            ret['id'] = id_

            return ret

        return collate_fn


if __name__ == "__main__":
    main()

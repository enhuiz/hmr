import os
import re
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def load_corpus(data_dir, typ):
    path = os.path.join(data_dir, 'annotations', 'tok_{}.csv'.format(typ))
    df = pd.read_csv(path)
    df.columns = ['id', 'latex']

    def printed_dir(x):
        return os.path.join(data_dir, 'features', 'printed', typ, x)

    def written_dir(x):
        return os.path.join(data_dir, 'features', 'written', typ, x)

    filenames = df['id'] + '.png'
    df['printed'] = filenames.apply(printed_dir)
    df['written'] = filenames.apply(written_dir)

    existence = df['printed'].apply(os.path.exists) \
        & df['written'].apply(os.path.exists)

    df = df[existence == True]

    df['latex'] = df['latex'].apply(lambda x: str(x).split(' '))
    df['len'] = df['latex'].apply(len)

    return df


class Vocab(object):
    def __init__(self, data_dir, unk='unk'):
        self.s2l = self.load_vocab(data_dir)
        self.l2s = {v: k for k, v in self.s2l.items()}
        self.unk = unk

    @staticmethod
    def load_vocab(data_dir):
        path = os.path.join(data_dir, 'annotations', 'vocab.csv')
        with open(path, 'r') as f:
            content = f.read().strip()
        vocab = content.split('\n')
        vocab = {w: i for i, w in enumerate(vocab)}
        return vocab

    def symbol2label(self, s):
        if s in self.s2l:
            return self.s2l[s]
        return len(self.s2l)  # unk

    def latex2labels(self, latex):
        return list(map(self.symbol2label, latex))

    def label2symbol(self, l):
        if l in self.l2s:
            return self.l2s[l]
        return self.unk

    def labels2latex(self, labels):
        return list(map(self.labels2latex, labels))

    def __len__(self):
        return len(self.s2l) + 1  # unk


class MathDataset(Dataset):
    def __init__(self, data_dir, typ, transform=None, pad_idx=2333, paired=True):
        self.vocab = Vocab(data_dir)
        self.paired = paired
        self.pad_idx = pad_idx

        corpus = load_corpus(data_dir, typ)
        self.max_len = max(corpus['len'])
        if not self.paired:
            # shuffle
            corpus['printed'] = corpus['printed'].sample(frac=1).values
        self.samples = corpus.to_dict('record')

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [1])
            ])
        else:
            self.transform = transform

    @staticmethod
    def load_pil(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __getitem__(self, idx):
        sample = dict(self.samples[idx])

        printed = self.transform(self.load_pil(sample['printed']))
        written = self.transform(self.load_pil(sample['written']))

        if self.paired:
            # only paired data have label
            latex = sample['latex']
            label = self.vocab.latex2labels(latex)
            label = label + [self.pad_idx] * (self.max_len - len(label))
            sample['label'] = np.array(label)
            sample['latex'] = ' '.join(latex)

        sample['printed'] = printed
        sample['written'] = written
        return sample

    def __len__(self):
        return len(self.samples)

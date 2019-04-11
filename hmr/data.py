import os
import re
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def load_symbol_list(s):
    s = eval(s)
    s = list(map(str, s))
    return s


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

    df['latex'] = df['latex'].apply(load_symbol_list)
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


class ICFHRDataset(Dataset):
    def __init__(self, data_dir, typ, transform=None, pad_idx=2333):
        self.vocab = Vocab(data_dir)
        corpus = load_corpus(data_dir, typ)
        self.max_len = max(corpus['len'])
        self.samples = corpus.to_dict('record')
        self.pad_idx = 2333

        if transform is None:
            self.transform = transforms.Compose([
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
        sample = self.samples[idx]
        label = self.vocab.latex2labels(sample['latex'])

        # loading
        printed = self.transform(self.load_pil(sample['printed']))
        written = self.transform(self.load_pil(sample['written']))

        # padding
        label = label + [self.pad_idx] * (self.max_len - len(label))

        sample['printed'] = printed
        sample['written'] = written
        sample['label'] = np.array(label)
        sample['latex'] = ' '.join(sample['latex'])

        return sample

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    dataset = ICFHRDataset('data/ICFHR', 'train')
    dl = DataLoader(dataset, batch_size=36)
    for d in dl:
        print(d)
        break

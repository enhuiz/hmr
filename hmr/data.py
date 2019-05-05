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
    def __init__(self, data_dir=None):
        if data_dir is not None:
            self.load(data_dir)

    def load(self, data_dir):
        path = os.path.join(data_dir, 'annotations', 'vocab.csv')

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


vocab = Vocab()


def create_samples(data_dir, typ, written=False):
    path = os.path.join(data_dir, 'annotations', '{}.csv'.format(typ))
    df = pd.read_csv(path)
    df.columns = ['id', 'annotation']

    filenames = df['id'] + '.png'

    def printed_dir(x):
        return os.path.join(data_dir, 'features', 'printed', typ, x)

    def written_dir(x):
        return os.path.join(data_dir, 'features', 'written', typ, x)

    if written:
        df['image'] = filenames.apply(written_dir)
    else:
        df['image'] = filenames.apply(printed_dir)

    existence = df['image'].apply(os.path.exists)
    df = df[existence]

    df['annotation'] = df['annotation'].apply(lambda s: s.strip())

    samples = df[['id', 'image', 'annotation']].to_dict('record')

    return samples


class MathDataset(Dataset):
    def __init__(self, data_dir, typ, transform, written=False):
        vocab.load(data_dir)
        self.transform = transform
        self.samples = create_samples(data_dir, typ, written)

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
        annotation = list(map(vocab.word2index, annotation))
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

    @staticmethod
    def collate_fn(batch):
        image = [sample['image'] for sample in batch]
        annotation = [sample['annotation'] for sample in batch]
        len_ = [sample['len'] for sample in batch]
        id_ = [sample['id'] for sample in batch]

        ret = {}
        pad_ix = vocab.word2index('<pad>')
        ret['image'] = torch.stack(image)
        ret['annotation'] = pad_sequence(annotation, False, pad_ix)
        ret['len'] = torch.tensor(len_)
        ret['id'] = id_

        return ret


def main():
    dataset = MathDataset('data/crohme', 'train',
                          transforms.ToTensor(), written=False)
    print('vocab len', len(vocab))
    dl = DataLoader(dataset, batch_size=8, shuffle=False,
                    collate_fn=MathDataset.collate_fn)
    for sample in dl:
        print({k: v.shape for k, v in sample.items() if hasattr(v, 'shape')})
        break


if __name__ == "__main__":
    main()

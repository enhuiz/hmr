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

        words.add('<s>')
        words.add('</s>')
        words.add('<pad>')
        words.add('<unk>')

        self.words = sorted(words)
        self.inv_words = {w: i for i, w in enumerate(self.words)}

    def word2index(self, w):
        return self.inv_words[w]

    def index2word(self, i):
        return self.words[i]

    def __len__(self):
        return len(self.words)


vocab = Vocab()


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

    df['caption'] = df['latex']

    return df


class MathDataset(Dataset):
    def __init__(self, data_dir, typ, transform, written=False):
        self.transform = transform

        vocab.load(data_dir)
        corpus = load_corpus(data_dir, typ)

        if written:
            corpus['image'] = corpus['written']
        else:
            corpus['image'] = corpus['printed']

        corpus = corpus[['id', 'image', 'caption']]
        self.samples = corpus.to_dict('record')

    @staticmethod
    def load_pil(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def process_caption(self, caption):
        caption = '<s> {} </s>'.format(caption).split(' ')
        caption = list(map(vocab.word2index, caption))
        caption = torch.tensor(caption)
        return caption

    def __getitem__(self, idx):
        sample = dict(self.samples[idx])
        image = self.transform(self.load_pil(sample['image']))
        caption = self.process_caption(sample['caption'])
        sample['image'] = image
        sample['caption'] = caption
        sample['len'] = len(caption)
        return sample

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def collate_fn(batch):
        image = [sample['image'] for sample in batch]
        caption = [sample['caption'] for sample in batch]
        len_ = [sample['len'] for sample in batch]
        id_ = [sample['id'] for sample in batch]

        ret = {}
        pad_ix = vocab.word2index('<pad>')
        ret['image'] = torch.stack(image)
        ret['caption'] = pad_sequence(caption, False, pad_ix)
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

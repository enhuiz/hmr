import os

vocab = None
dict_ = None


def init_vocab(data_dir):
    global vocab, dict_

    path = os.path.join(data_dir, 'annotations', 'vocab.csv')
    with open(path, 'r') as f:
        content = f.read().strip()
    vocab = content.split('\n')
    vocab = set(vocab)
    vocab.add('<s>')
    vocab.add('</s>')
    vocab.add('<pad>')
    vocab = sorted(vocab)
    dict_ = {w: i for i, w in enumerate(vocab)}


def word2index(w):
    return dict_[w]


def index2word(i):
    return vocab[i]


def size():
    return len(vocab)

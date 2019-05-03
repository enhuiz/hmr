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
    vocab.add('<unk>')
    vocab = sorted(vocab)
    dict_ = {w: i for i, w in enumerate(vocab)}


def word2index(w):
    if w in dict_:
        return dict_[w]
    else:
        return dict_['<unk>']


def index2word(i):
    return vocab[i]


def size():
    return len(vocab)

import os
import argparse
import mnist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import tqdm

from PIL import Image, ImageOps

from tokenize_latex import tokenize

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from hmr.data.render import latex2png


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir')
    args = parser.parse_args()
    return args


def split(data, ratio):
    ratio = np.clip(ratio, 0, 1)
    n = int(len(data) * ratio)
    return data[:n], data[n:]


def create(data, out_dir, typ):

    def create_written():
        print('Creating written images')
        written_dir = os.path.join(out_dir, 'features', 'written', typ)
        os.makedirs(written_dir, exist_ok=True)
        for i in tqdm.tqdm(range(len(data)), total=len(data)):
            id_ = '{}_{}'.format(typ, i)
            img = Image.fromarray(data[i][0])
            img = ImageOps.invert(img)  # to white background
            img.save(os.path.join(written_dir, '{}.png'.format(id_)))

    def create_annotation():
        print('Creating annotation')
        path = os.path.join(out_dir, 'annotations', '{}.csv'.format(typ))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pairs = []
        for i in tqdm.tqdm(range(len(data)), total=len(data)):
            id_ = '{}_{}'.format(typ, i)
            annotation = data[i][1]
            pairs.append([id_, annotation])
        df = pd.DataFrame(pairs)
        df.to_csv(path, index=None, header=None)

    def create_tok_annotation():
        print('Creating tokenized annotation')
        path = os.path.join(out_dir, 'annotations', 'tok_{}.csv'.format(typ))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pairs = []
        for i in tqdm.tqdm(range(len(data)), total=len(data)):
            id_ = '{}_{}'.format(typ, i)
            annotation = tokenize(str(data[i][1]))
            pairs.append([id_, annotation])
        df = pd.DataFrame(pairs)
        df.to_csv(path, index=None, header=None)

    def create_printed():
        print('Creating printed images')
        csv_path = os.path.join(out_dir, 'annotations', '{}.csv'.format(typ))
        printed_dir = os.path.join(out_dir, 'features', 'printed', typ)
        os.makedirs(printed_dir, exist_ok=True)
        os.system('node scripts/data/latex2png.js {} {} '
                  '--size 28 28 --pad 0'.format(csv_path, printed_dir))

    create_written()
    create_annotation()
    create_tok_annotation()
    create_printed()  # must be done after annotation


def main():
    args = get_args()

    dtrain = list(zip(mnist.train_images(), mnist.train_labels()))
    dtrain, dval = split(dtrain, 0.9)

    dtest = list(zip(mnist.test_images(), mnist.test_labels()))

    create(dtrain, args.out_dir, 'train')
    create(dval, args.out_dir, 'val')
    create(dtest, args.out_dir, 'test')


if __name__ == "__main__":
    main()

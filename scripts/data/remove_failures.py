import os
import argparse
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()
    return args


def remove_failures(path):
    df = pd.read_csv(path, names=['id', 'annotation'])
    prev_len = len(df)
    if prev_len == 0:
        return

    typ = path.split('/')[-1].split('.')[0]
    features_dir = os.path.join(*path.split('/')[:-2], 'features')
    for style in ['printed', 'written']:
        image_path = os.path.join(features_dir, style, typ, '{}.png')
        df['image'] = df['id'].apply(image_path.format)
        df = df[df['image'].apply(os.path.exists)]

    df[['id', 'annotation']].to_csv(path, header=None, index=None)
    cur_len = len(df)

    if prev_len != cur_len:
        print('{} failures removed.'.format(prev_len - cur_len))


if __name__ == "__main__":
    args = get_args()
    remove_failures(args.path)

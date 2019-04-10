import re
import argparse

import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_paths', nargs='+')
    parser.add_argument('out_path')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    tokens = set()
    for csv_path in args.csv_paths:
        try:
            df = pd.read_csv(csv_path, header=None)
        except pd.errors.EmptyDataError as e:
            continue
        for sentence in df[1].values:
            sentence = map(str, eval(sentence))
            tokens.update(sentence)
    tokens = sorted(tokens)
    with open(args.out_path, 'w') as f:
        f.write('\n'.join(tokens))


if __name__ == "__main__":
    main()

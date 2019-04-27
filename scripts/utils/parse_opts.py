#!/usr/bin/env python3

import json
import argparse


def parse_key(k):
    k = k.replace('_', '-')
    return k


def parse_value(v):
    if isinstance(v, list):
        v = ' '.join(map(str, v))
    return v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json')
    args = parser.parse_args()

    with open(args.json, 'r') as f:
        d = json.load(f)

    opts = ['--{} {}'.format(parse_key(k), parse_value(d[k])) for k in d]
    opts = ' '.join(opts)
    print(opts, end="")


if __name__ == "__main__":
    main()

import re
import argparse

import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_csv')
    parser.add_argument('out_csv')
    args = parser.parse_args()
    return args


def tokenize(latex, ignore_types=['WS']):
    _patterns = [
        r'(?P<NUM>[0-9]+\.?[0-9]*)',
        r'(?P<MM_B_ALIGN>\\begin\{align\*?\})',
        r'(?P<MM_E_ALIGN>\\end\{align\*?\})',
        r'(?P<MM_DMATH>\\\[|\$\$|\\\])',
        r'(?P<MM_BILMATH>^\$)',
        r'(?P<MM_EINMATH>\$$)',
        r'(?P<BEG>\\begin\{[^}]*\})',
        r'(?P<END>\\end\{[^}]*\})',
        r'(?P<NL>\\\\)',
        r'(?P<MP>\\[{}|])',
        r'(?P<TEXT>\\text\w*)',
        r'(?P<MATHFONT>\\math\w*\{[^}]*})',
        r'(?P<GB>\{)',
        r'(?P<GE>\})',
        r'(?P<ROOT>\\sqrt\[([^]]*)\])',
        r'(?P<SQRT>\\sqrt)',
        r'(?P<SUB>\_)',
        r'(?P<SUP>\^)',
        r'(?P<LEFT>\\left(?!\w))',
        r'(?P<RIGHT>\\right(?!\w))',
        r'(?P<FRAC>\\frac)',
        r'(?P<BIN>\\binom)',
        r'(?P<OVERUNDER>\\over[a-z]*|\\under[a-z]*)',
        r'(?P<COM>\\[A-Za-z]+|\\[,:;\s])',
        r'(?P<WS>\s+)',
        r'(?P<AMP>&)',
        r'(?P<SYMB>.)'
    ]
    patterns = re.compile('|'.join(_patterns))
    sn = patterns.scanner(latex)
    tokens = []
    for m in iter(sn.match, None):
        if m.lastgroup not in ignore_types:
            tokens.append(m.group())
    return tokens


def main():
    args = get_args()
    try:
        df = pd.read_csv(args.in_csv, header=None)
        df[1] = df[1].apply(tokenize)
        df.to_csv(args.out_csv)
    except pd.errors.EmptyDataError as e:
        print('Warning: {} is empty!'.format(args.in_csv))
        with open(args.out_csv, 'w'):
            pass


if __name__ == "__main__":
    main()

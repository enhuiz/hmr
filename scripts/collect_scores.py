import os
import argparse
import sys
import glob
import pandas as pd


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    opts = parser.parse_args()
    return opts


def pandas_df_to_markdown_table(df):
    from IPython.display import Markdown, display
    fmt = ['---' for i in range(len(df.columns))]
    df_fmt = pd.DataFrame([fmt], columns=df.columns)
    df_formatted = pd.concat([df_fmt, df])
    return df_formatted.to_csv(sep="|", index=None)


def main():
    opts = get_opts()
    records = []
    for path in glob.glob(os.path.join(opts.dir, '**', 'scores.txt'), recursive=True):
        with open(path, 'r') as f:
            d = eval(f.read())
        d['name'] = path.lstrip(opts.dir).split('/')[0]
        d['style'] = path.lstrip(opts.dir).split('/')[1]
        d['part'] = path.lstrip(opts.dir).split('/')[2]
        records.append(d)

    df = pd.DataFrame(records)
    df = df[['style', 'name', 'part', 'BLEU-4', 'Edit']].sort_values('style')
    print(pandas_df_to_markdown_table(df.round(2)))


if __name__ == "__main__":
    main()

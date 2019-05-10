import os
import argparse
import glob
import pandas as pd


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('out_dir', type=str)
    opts = parser.parse_args()
    return opts


def split_formulas(path):
    df = pd.read_csv(path)
    print(df)


def get_annotation_writer(src_dir):

    def annotation_writer(out_dir):
        for path in glob.glob(src_dir + '/tok_*.csv'):
            part = path.split('tok_')[-1].split('.csv')[0]
            part = 'val' if part == 'dev' else part
            formulas_path = os.path.join(
                out_dir, '{}.formulas.norm.txt'.format(part))
            matching_path = os.path.join(
                out_dir, '{}.matching.txt'.format(part))
            try:
                df = pd.read_csv(path, header=None, names=['image', 'formula'])
                df['index'] = df.index
                df['formula'].to_csv(formulas_path, header=False, index=None)
                df[['image', 'index']].to_csv(
                    matching_path, header=False, index=None, sep=' ')
            except Exception as e:
                print(e)
                os.system('touch {} {}'.format(formulas_path, matching_path))

        os.system('cp {}/vocab.csv {}/vocab.txt'.format(src_dir, out_dir))

    return annotation_writer


def main():
    opts = get_opts()
    os.makedirs(opts.out_dir, exist_ok=True)
    aw = get_annotation_writer(os.path.join(opts.data_dir, 'annotations'))

    for img_dir in glob.glob(os.path.join(opts.data_dir, 'features/*')):
        mode = os.path.basename(img_dir)
        out_dir = os.path.join(opts.out_dir, mode)
        os.makedirs(out_dir, exist_ok=True)
        for part in os.listdir(img_dir):
            mapped_typ = 'images_val' if part == 'dev' else 'images_{}'.format(
                part)
            os.system('cp -r {} {}'.format(os.path.join(img_dir, part),
                                           os.path.join(out_dir, mapped_typ)))
        aw(out_dir)


if __name__ == "__main__":
    main()

dst=$1

if [ -z "$dst" ]; then
    echo 'Please give an output dir.'
    exit 1
fi

mkdir -p $dst/annotations/raw

wget https://raw.githubusercontent.com/guillaumegenthial/im2latex/master/data/test.formulas.norm.txt -P $dst/annotations/raw
wget https://raw.githubusercontent.com/guillaumegenthial/im2latex/master/data/train.formulas.norm.txt -P $dst/annotations/raw
wget https://raw.githubusercontent.com/guillaumegenthial/im2latex/master/data/val.formulas.norm.txt -P $dst/annotations/raw

python3 -c "
import pandas as pd

def process(name, in_path, out_path):
    with open(in_path, 'r') as f:
        samples = f.read().strip().split('\n')
    df = pd.DataFrame(samples, columns=['math'])
    df['id'] = list(map(lambda x: '{}_{}'.format(name, x), df.index))
    df = df[['id', 'math']]
    df.to_csv(out_path, header=None, index=None)

process('im2latex_train', '$dst/annotations/raw/train.formulas.norm.txt', '$dst/annotations/train.csv')
process('im2latex_test', '$dst/annotations/raw/test.formulas.norm.txt', '$dst/annotations/test.csv')
process('im2latex_dev', '$dst/annotations/raw/val.formulas.norm.txt', '$dst/annotations/dev.csv')
"

python3 scripts/data/tokenize_latex.py $dst/annotations/{train,train}.csv
python3 scripts/data/tokenize_latex.py $dst/annotations/{dev,dev}.csv
python3 scripts/data/tokenize_latex.py $dst/annotations/{test,test}.csv

python3 scripts/data/build_vocab.py $dst/annotations/{train,vocab}.csv

node scripts/data/latex2png.js $dst/annotations/train.csv $dst/features/printed/train
node scripts/data/latex2png.js $dst/annotations/dev.csv $dst/features/printed/dev
node scripts/data/latex2png.js $dst/annotations/test.csv $dst/features/printed/test
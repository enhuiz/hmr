# Handwritten Mathematical Recognition

## Steps

### 1. Preprocessing

This is for the data preprocessing. If you already have the dataset like the following, you can skip it.

```
── {dataset-name}       <- dataset name e.g. mnist, crohme or im2latex
   ├── annotations      <- math formular latex code
   │   ├── dev.csv      <- csv files, in format of id,formula
   │   ├── test.csv
   │   ├── train.csv
   |   └── vocab.csv    <- vocab build by scripts/data/build_vocab.py
   └── features         <- math formular images
       ├── printed      <- printed version
       │   ├── dev
       │   ├── test
       │   └── train
       └── written      <- written version
           ├── dev
           ├── test
           └── train
```

1. Install [nodejs](https://nodejs.org/en/) and [npmjs](https://www.npmjs.com/), and then run `npm install` under the project root. 

```
$ sudo apt install nodejs npm
$ npm install
```

2. Run the following script to build crohme dataset

```
$ scripts/data/create_crohme19.sh data/crohme
```

3. Run the following script to build mnist dataset

```
$ scripts/data/create_mnist.py data/mnist
```

4. Run the following script to build im2latex dataset

```
$ scripts/data/create_im2latex.py data/im2latex
```

### 2. Train Style Transfers

Train style transfers on mnist or crohme.

```
python3 train_gan.py config/cyclegan/mnist.yml
```

### 3. Train Seq2seq Model
 

Train seq2seq on crohme or im2latex.

```
python3 train_gan.py config/seq2seq/crohme.yml
```
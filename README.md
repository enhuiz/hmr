# Handwritten Mathematical Recognition

## Steps

### 1. Download Data 

Donwload ICFHR from [here](http://www.isical.ac.in/~crohme/ICFHR_package.zip) and put it into `data/raw/ICFHR/` folder.
Donwload MNIST from [here](http://yann.lecun.com/exdb/mnist/) and put it into `data/raw/MNIST/` folder.

### 2. Preprocess Data

Preprocess the data into `data/` to generate directories which has the following structure:

```bash
── ICFHR 
   ├── annotations      <- math formular latex code
   │   ├── dev.csv      <- csv files, in format: id,latex-code
   │   ├── test.csv
   │   └── train.csv
   └── features         <- math formular images
       ├── printed      <- printed version, provided by ICFHR as latex, should be rendered by latex.
       │   ├── dev
       │   ├── test
       │   └── train
       └── written      <- written version, provided by ICFHR as trajectory, should be rendered by some scripts.
           ├── dev
           ├── test
           └── train
── MNIST 
   ├── annotations      <- digits
   │   ├── dev.csv      <- csv files, in format: image-path,digit
   │   ├── test.csv
   │   └── train.csv
   └── features         <- math formular images
       ├── printed      <- printed version, rendered by latex
       │   ├── dev
       │   ├── test
       │   └── train
       └── written      <- written version, provide by the MNIST as digit, should be rendered by latex.
           ├── dev
           ├── test
           └── train
```

Note that samples are named by an integer id, different samples in the same dataset should have different ids. 
For example, if you have totally 200 samples in ICFHR, you design to put 180 of them to train and 10 dev 10 test, then you need to do it in the following way:

Put the {printed,written} images 0.png-179.png to `ICFHR/features/{printed,written}/train`, 
then put the {printed,written} images 180.png-189.png to `ICFHR/features/{printed,written}/dev`,
and the {printed,written} 190.png-199.png to `ICFHR/features/{printed,written}/test`.

Note that the ids should be consistant, which means printed/train/0.png and written/train/0.png should be the SAME sample (i.e. the same math formular).


### 3. Train Style Transfer  

Train a style transferer using written->printed pairs from both ICFHR and MNIST dataset, note that only samples in `train/` and `train.csv` should be used during training. 

### 4. Train Seq2seq Model 

Train an encoder decoder network using ICFHR printed->annotaions pairs, note that only samples in `train/` and `train.csv` should be used during training.


Both step 3 and step 4 may follow the following directory structure:

```bash
├── hmr      <- put your .py here, these scripts used as library, which are not executable directly, e.g. `data_utils.py`, `seq2seq.py`, etc.
├── scripts  <- put your .py or .sh scripts, these scripts should be able to executable directly, e.g. `train_gan.sh`, `process_data.py`, etc.
```

Feel free to modify this file.

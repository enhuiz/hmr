#!/bin/bash

device=$1

[ -z $device ] && device=cpu
[ -z $stage ] && stage=0

name=cyclegan
mean=0.5

# train on mnist

lr=1e-4
batch_size=8
epochs=10

python3 -u scripts/train_gan.py \
    --device $device \
    --data-dir data/mnist \
    --batch-size $batch_size \
    --epochs $epochs \
    --name $name/mnist \
    --upernet true\
    --lr $lr \
    --mean "$mean" || exit 1

# train on crohme

lr=1e-4
epochs=10
batch_size=4
model=$(ls -1v checkpoints/$name/mnist/*.pth | tail -1)

python3 -u scripts/train_gan.py \
    --device $device \
    --model $model \
    --data-dir data/crohme \
    --batch-size $batch_size \
    --epochs $epochs \
    --name $name/crohme \
    --lr $lr \
    --mean "$mean" || exit 1

#!/bin/bash

device=$1
name=$2

[ -z $device ] && device=cpu
[ -z $name ] && name=cyclegan

model=UPerNetCycleGAN #UNetCycleGAN
name=cyclegan
mean=0.5

# train on mnist

lr=1e-4
batch_size=16
epochs=10

python3 -u scripts/train_gan.py \
    --device $device \
    --data-dir data/mnist \
    --batch-size $batch_size \
    --epochs $epochs \
    --name $name/mnist \
    --model $model \
    --lr $lr \
    --mean "$mean" || exit 1

# train on crohme

lr=1e-4
epochs=10
batch_size=8
model=$(ls -1v checkpoints/$name/mnist/*.pth | tail -1)

python3 -u scripts/train_gan.py \
    --device $device \
    --model $model \
    --data-dir data/crohme \
    --batch-size $batch_size \
    --epochs $epochs \
    --name $name/crohme \
    --model $model \
    --lr $lr \
    --mean "$mean" || exit 1

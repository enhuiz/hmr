#!/bin/bash

device=$1
name=$2

[ -z $device ] && device=cpu
[ -z $name ] && name=cyclegan

model=UPerNetCycleGAN
mean=0.5

# train mixed

lr=5e-4
batch_size=16
epochs=1

python3 -u scripts/train_gan.py \
    --device $device \
    --data-dir data/mnist data/crohme \
    --batch-size $batch_size \
    --epochs $epochs \
    --name $name/mixed \
    --model $model \
    --lr $lr \
    --mean "$mean" || exit 1

# train on crohme

lr=2e-4
epochs=5
batch_size=16
model=$(ls -1v checkpoints/$name/mixed/*.pth | tail -1)

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

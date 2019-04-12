#!/bin/bash

device=$1

[ -z $device ] && device=cpu
[ -z $stage ] && stage=0

model=dilated_resnet18_upernet
data_dir=data/ICFHR/
out_dir=exp/gan2/
batch_size=2
epochs=30
lr=1e-5
mean=0.5

python3 -u scripts/train_gan.py \
    --device $device \
    --data-dir $data_dir \
    --batch-size $batch_size \
    --epochs $epochs \
    --out-dir $out_dir \
    --lr $lr \
    --mean "$mean" || exit 1

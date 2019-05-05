#!/bin/bash

out_dir=$1

for typ in dev train; do
    python3 -u scripts/forward_cycle_gan.py \
        --data-dir data/crohme \
        --model ckpt/ResNetCycleGAN/crohme/25.pth \
        --type $typ \
        --out-dir $out_dir/$typ \
        --device cpu
done

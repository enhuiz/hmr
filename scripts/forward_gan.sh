#!/bin/bash

model=$1

for part in dev; do
    python3 -u scripts/forward_gan.py \
        --data-dir data/crohme \
        --model $model \
        --part $part \
        --out-dir data/crohme/features/ \
        --device cuda
done

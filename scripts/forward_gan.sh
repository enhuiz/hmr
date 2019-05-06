#!/bin/bash

model=$1

for typ in dev; do
    python3 -u scripts/forward_gan.py \
        --data-dir data/crohme \
        --model $model \
        --type $typ \
        --out-dir data/crohme/features/fake/$typ \
        --device cuda
done

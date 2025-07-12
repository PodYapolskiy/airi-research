#!/bin/bash

uv run src/train_personality.py \
    --data-dir /home/solan/repos/airi-research/data \
    --batch-size 514 \
    --num-workers 4 \
    --device cuda:0 \
    --epochs 250 \
    --lr 0.001 \
    --trait "Conscientiousness" \
    --with-meta \
    --with-video \
    --with-audio \
    --with-text \
    --fusion early \
    --train-concated

#!/bin/bash

uv run src/train_performance.py \
    --data-dir /home/HDD12TB/datasets/images/emotions/ACMMM25/AVI/AVI_Challenge_dataset \
    --batch-size 450 \
    --num-workers 4 \
    --device cuda \
    --epochs 100 \
    --lr 0.001 \
    --trait "..." \
    --with-meta \
    --with-video \
    --with-audio \
    --with-text \
    --fusion "early"

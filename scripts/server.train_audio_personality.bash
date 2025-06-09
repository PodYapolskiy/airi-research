#!/bin/bash

uv run src/train_audio_personality.py \
    --data-dir /home/HDD12TB/datasets/images/emotions/ACMMM25/AVI/AVI_Challenge_dataset \
    --batch-size 128 \
    --num-workers 4 \
    --device cuda \
    --epochs 25 \
    --lr 0.001 \
    --trait "Conscientiousness" \
    --only-dim 512

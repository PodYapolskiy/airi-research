#!/bin/bash

uv run src/preprocess.py \
    --data-dir /home/HDD12TB/datasets/images/emotions/ACMMM25/AVI/AVI_Challenge_dataset \
    --video-model-device cuda:0 \
    --preprocess-video

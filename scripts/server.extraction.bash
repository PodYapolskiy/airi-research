#!/bin/bash

uv run src/extraction.py \
    --data-dir /home/HDD12TB/datasets/images/emotions/ACMMM25/AVI/AVI_Challenge_dataset \
    --train-dir train_data \
    --val-dir val_data \
    --preprocessed-train-dir preprocessed_train_data \
    --preprocessed-val-dir preprocessed_val_data \
    --train-csv train_data.csv \
    --val-csv val_data.csv \
    --extract-video 0 \
    --extract-audio 1 \
    --extract-text 1 \
    --video-model-device cpu \
    --text-model-device cpu \

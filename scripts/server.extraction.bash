#!/bin/bash

uv run src/extraction.py \
    --data-dir /home/HDD12TB/datasets/images/emotions/ACMMM25/AVI/AVI_Challenge_dataset \
    --train-dir train_data \
    --val-dir val_data \
    --preprocessed-train-dir preprocessed_train_data \
    --preprocessed-val-dir preprocessed_val_data \
    --train-csv train_data.csv \
    --val-csv val_data.csv \
    --video-model-device cpu \
    --text-model-device cpu \
    --extract-text \
    --extract-audio \
    # --extract-video \

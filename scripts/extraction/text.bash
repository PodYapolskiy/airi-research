#!/bin/bash

uv run src/extraction.py \
    --data-dir /home/HDD12TB/datasets/images/emotions/ACMMM25/AVI/AVI_Challenge_dataset \
    --text-model-device cuda \
    --extract-text

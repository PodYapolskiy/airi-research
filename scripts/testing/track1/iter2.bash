#!/bin/bash

uv run src/test_personality.py \
    --data-dir /home/HDD12TB/datasets/images/emotions/ACMMM25/AVI/AVI_Challenge_dataset \
    --trait Honesty-Humility \
    --with-meta \
    --with-audio \
    --with-text \
    --fusion early

uv run src/test_personality.py \
    --data-dir /home/HDD12TB/datasets/images/emotions/ACMMM25/AVI/AVI_Challenge_dataset \
    --trait Extraversion \
    --with-meta \
    --with-audio \
    --with-text \
    --fusion early

uv run src/test_personality.py \
    --data-dir /home/HDD12TB/datasets/images/emotions/ACMMM25/AVI/AVI_Challenge_dataset \
    --trait Agreeableness \
    --with-meta \
    --with-audio \
    --with-text \
    --fusion early

uv run src/test_personality.py \
    --data-dir /home/HDD12TB/datasets/images/emotions/ACMMM25/AVI/AVI_Challenge_dataset \
    --trait Conscientiousness \
    --with-meta \
    --with-audio \
    --with-text \
    --fusion early

uv run src/submit_personality.py \
    --data-dir /home/HDD12TB/datasets/images/emotions/ACMMM25/AVI/AVI_Challenge_dataset \

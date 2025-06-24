#!/bin/bash

traits=("Integrity" "Collegiality" "Social_versatility" "Development_orientation" "Hireability")

modalities_combinations=(
    "--with-meta"
    "--with-video"
    "--with-audio"
    "--with-text"
    "--with-meta --with-video"
    "--with-meta --with-audio"
    "--with-meta --with-text"
    "--with-video --with-audio"
    "--with-video --with-text"
    "--with-audio --with-text"
    "--with-meta --with-video --with-audio"
    "--with-meta --with-video --with-text"
    "--with-meta --with-audio --with-text"
    "--with-video --with-audio --with-text"
    "--with-meta --with-video --with-audio --with-text"
)

for trait in "${traits[@]}"; do
    (
        for modalities in "${modalities_combinations[@]}"; do
            uv run src/train_performance.py \
                --data-dir /home/HDD12TB/datasets/images/emotions/ACMMM25/AVI/AVI_Challenge_dataset \
                --batch-size 450 \
                --num-workers 4 \
                --device cuda \
                --epochs 150 \
                --lr 0.001 \
                --fusion late \
                --trait "$trait" \
                $modalities
        done
    ) &
done

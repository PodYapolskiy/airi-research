#!/bin/bash

uv run src/test_personality.py \
    --data-dir /home/solan/repos/airi-research/data \
    --trait Honesty-Humility \
    --with-meta \
    --with-video \
    --with-audio \
    --with-text \
    --fusion early

uv run src/test_personality.py \
    --data-dir /home/solan/repos/airi-research/data \
    --trait Extraversion \
    --with-meta \
    --with-video \
    --with-audio \
    --with-text \
    --fusion early

uv run src/test_personality.py \
    --data-dir /home/solan/repos/airi-research/data \
    --trait Agreeableness \
    --with-meta \
    --with-video \
    --with-audio \
    --with-text \
    --fusion early

uv run src/test_personality.py \
    --data-dir /home/solan/repos/airi-research/data \
    --trait Conscientiousness \
    --with-meta \
    --with-video \
    --with-audio \
    --with-text \
    --fusion early

uv run src/submit_personality.py \
    --data-dir /home/solan/repos/airi-research/data \

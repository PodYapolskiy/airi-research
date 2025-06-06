# AIRI Research

## AVI Challenge

...

## Metrics

Dummy - ridge regression on age, work_experience, gender, education

<!-- Baseline - -->

#### Personality

| Metric  | Label             | Dummy  | Baseline |
| :-----: | :---------------- | :----: | :------: |
|   MSE   | Honesty-Humility  | 0.185  |          |
|         | Extraversion      | 0.282  |          |
|         | Agreeableness     | 0.216  |          |
|         | Conscientiousness | 0.179  |          |
| $ R^2 $ | Honesty-Humility  | 0.022  |          |
|         | Extraversion      | -0.001 |          |
|         | Agreeableness     | 0.014  |          |
|         | Conscientiousness | 0.052  |          |

#### Performance

| Metric  | Label                   | Dummy | Baseline |
| :-----: | :---------------------- | :---: | :------: |
|   MSE   | Integrity               | 0.204 |          |
|         | Collegiality            | 0.292 |          |
|         | Social_versatility      | 0.286 |          |
|         | Development_orientation | 0.227 |          |
|         | Hireability             | 0.350 |          |
| $ R^2 $ | Integrity               | 0.063 |          |
|         | Collegiality            | 0.097 |          |
|         | Social_versatility      | 0.100 |          |
|         | Development_orientation | 0.039 |          |
|         | Hireability             | 0.111 |          |

## Modalities Extraction

For $ \forall $ \*.mp4:

- \*.mp4 (extracted faces without sound)
- \*.wav
- \*.txt

```bash
uv run src/extraction.py
```

## Preprocessing

```bash
uv run src/preprocess.py \
#    --data-dir data \
#    --train-dir train_data \
#    --val_data val_data \
#    --preprocessed-train-dir preprocessed_train_data \
#    --preprocessed-val-dir preprocessed_val_data \
#    --train-csv train_data.csv \
#    --val-csv val_data.csv \
#    --image-size 224 \
#    --min-face-size 50 \
#    --custom-preprocess 0 \
```

## Links

https://github.com/EvelynFan/AWESOME-FER?tab=readme-ov-file

https://ieeexplore.ieee.org/document/9649076
https://ieeexplore.ieee.org/document/9815154
https://ieeexplore.ieee.org/document/9896386

# AIRI Research

## AVI Challenge

...

## Metrics

- [Only Meta](./notebooks/dummy.ipynb) - ridge regression on `["age", "work_experience", "gender", "education"]`
- [Only Video](.) - ...
- [Only Audio](./src/train_audio_personality.py) - OnlyNet
- [Only Text](./src/train_text_personality.py) - OnlyNet

#### Personality

|  Metric   | Label             | Only Meta | Only Video | Only Audio | Only Text |
| :-------: | :---------------- | :-------: | :--------: | :--------: | :-------: |
|    MSE    | Honesty-Humility  |   0.185   |            |   0.190    |   0.180   |
|           | Extraversion      |   0.282   |            |   0.278    |   0.219   |
|           | Agreeableness     |   0.216   |            |   0.218    |   0.225   |
|           | Conscientiousness |   0.179   |            |   0.189    |   0.215   |
| $` R^2 `$ | Honesty-Humility  |   0.022   |            |   -0.002   |   0.051   |
|           | Extraversion      |  -0.001   |            |   0.013    |   0.224   |
|           | Agreeableness     |   0.014   |            |   0.002    |  -0.030   |
|           | Conscientiousness |   0.052   |            |   -0.001   |  -0.140   |

#### Performance

|  Metric   | Label                   | Only Meta | Only Video | Only Audio | Only Text |
| :-------: | :---------------------- | :-------: | :--------: | :--------: | :-------: |
|    MSE    | Integrity               |   0.204   |            |            |           |
|           | Collegiality            |   0.292   |            |            |           |
|           | Social_versatility      |   0.286   |            |            |           |
|           | Development_orientation |   0.227   |            |            |           |
|           | Hireability             |   0.350   |            |            |           |
| $` R^2 `$ | Integrity               |   0.063   |            |            |           |
|           | Collegiality            |   0.097   |            |            |           |
|           | Social_versatility      |   0.100   |            |            |           |
|           | Development_orientation |   0.039   |            |            |           |
|           | Hireability             |   0.111   |            |            |           |

## Setup

```bash
uv sync --index-strategy unsafe-best-match
```

## Modalities Extraction

For $` \forall `$ \*.mp4:

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

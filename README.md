# AIRI Research

## AVI Challenge

...

## Metrics

- MSE - the less, better
- $`R^2`$ - closer to 1 is better

Models:

- [Only Meta](./notebooks/dummy.ipynb) - ridge regression on `["age", "work_experience", "gender", "education"]`
- [Only Video](./src/train_video_personality.py) - OnlyNet on `[embedding (1280)]`
- [Only Audio](./src/train_audio_personality.py) - OnlyNet on `[embedding (512)]`
- [Only Text](./src/train_text_personality.py) - OnlyNet on `[embedding (1024)]`

#### Personality

|  Metric   | Label             | Only Meta | Only Video | Only Audio | Only Text | Meta Text (late fusion) | Meta Video Audio Text (late fusion) |
| :-------: | :---------------- | :-------: | :--------: | :--------: | :-------: | :---------------------: | :---------------------------------: |
|    MSE    | Honesty-Humility  |   0.185   |   0.185    |   0.190    | **0.180** |          0.201          |                0.200                |
|           | Extraversion      |   0.282   |   0.288    |   0.278    | **0.219** |          0.242          |                0.244                |
|           | Agreeableness     | **0.216** |   0.252    |   0.218    |   0.225   |          0.217          |                0.224                |
|           | Conscientiousness | **0.179** |   0.234    |   0.189    |   0.215   |          0.224          |                0.221                |
|           |                   |           |            |            |           |                         |                                     |
| $` R^2 `$ | Honesty-Humility  |   0.022   |   0.023    |   -0.002   | **0.051** |         -0.063          |               -0.057                |
|           | Extraversion      |  -0.001   |   -0.021   |   0.013    | **0.224** |          0.141          |                0.133                |
|           | Agreeableness     | **0.014** |   -0.151   |   0.002    |  -0.030   |          0.009          |               -0.022                |
|           | Conscientiousness | **0.052** |   -0.240   |   -0.001   |  -0.140   |         -0.191          |               -0.176                |

#### Performance

|  Metric   | Label                   |   Meta    | Video | Audio |  Text  | Meta Text (late fusion) | Meta Video Audio Text (late fusion) |
| :-------: | :---------------------- | :-------: | :---: | :---: | :----: | :---------------------: | :---------------------------------: |
|    MSE    | Integrity               | **0.204** | 0.216 | 0.216 | 0.208  |          0.212          |                0.211                |
|           | Collegiality            | **0.292** | 0.314 | 0.320 | 0.312  |          0.349          |                0.324                |
|           | Social_versatility      | **0.286** | 0.298 | 0.312 | 0.303  |          0.330          |                0.300                |
|           | Development_orientation |   0.227   | 0.225 | 0.234 | 0.242  |          0.243          |              **0.223**              |
|           | Hireability             | **0.350** | 0.374 | 0.390 | 0.365  |          0.395          |                0.363                |
|           |                         |           |       |       |        |                         |                                     |
| $` R^2 `$ | Integrity               | **0.063** | 0.009 | 0.007 | 0.044  |          0.029          |                0.031                |
|           | Collegiality            | **0.097** | 0.030 | 0.010 | 0.035  |         -0.077          |                0.000                |
|           | Social_versatility      | **0.100** | 0.060 | 0.016 | 0.044  |         -0.039          |                0.580                |
|           | Development_orientation |   0.039   | 0.046 | 0.008 | -0.026 |         -0.030          |              **0.054**              |
|           | Hireability             | **0.111** | 0.051 | 0.010 | 0.073  |         -0.002          |                0.079                |

## Setup

```bash
uv sync
```

## Extraction

For $` \forall `$ \*.mp4:

- \*.mp4 (extracted faces without sound)
- \*.wav
- \*.txt

```bash
uv run src/extraction.py
```

## Preprocessing

```bash
uv run src/preprocess.py
```

## Links

https://github.com/EvelynFan/AWESOME-FER?tab=readme-ov-file

https://ieeexplore.ieee.org/document/9649076
https://ieeexplore.ieee.org/document/9815154
https://ieeexplore.ieee.org/document/9896386

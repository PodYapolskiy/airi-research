import argparse
import os
from pathlib import Path
import requests
import mlflow
import pandas as pd

MLFLOW_URI = "http://localhost:5000"


def ensure_mlflow() -> None:
    try:
        response = requests.head(MLFLOW_URI)
        response.raise_for_status()
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError):
        raise RuntimeError(
            f"MLflow server is not running at {MLFLOW_URI}. Please start it with `mlflow server`."
        )

    mlflow.set_tracking_uri(uri=MLFLOW_URI)


def ensure_paths(data_dir_path: str, args: argparse.Namespace) -> None:
    DATA_DIR_PATH = Path(data_dir_path)
    TRAIN_DIR_PATH = DATA_DIR_PATH / args.train_dir
    VAL_DIR_PATH = DATA_DIR_PATH / args.val_dir
    PREPROCESSED_TRAIN_DIR_PATH = DATA_DIR_PATH / args.preprocessed_train_dir
    PREPROCESSED_VAL_DIR_PATH = DATA_DIR_PATH / args.preprocessed_val_dir

    os.makedirs(TRAIN_DIR_PATH, exist_ok=True)
    os.makedirs(PREPROCESSED_TRAIN_DIR_PATH, exist_ok=True)
    os.makedirs(PREPROCESSED_TRAIN_DIR_PATH / "video", exist_ok=True)
    os.makedirs(PREPROCESSED_TRAIN_DIR_PATH / "audio", exist_ok=True)
    os.makedirs(PREPROCESSED_TRAIN_DIR_PATH / "text", exist_ok=True)

    os.makedirs(VAL_DIR_PATH, exist_ok=True)
    os.makedirs(PREPROCESSED_VAL_DIR_PATH, exist_ok=True)
    os.makedirs(PREPROCESSED_VAL_DIR_PATH / "video", exist_ok=True)
    os.makedirs(PREPROCESSED_VAL_DIR_PATH / "audio", exist_ok=True)
    os.makedirs(PREPROCESSED_VAL_DIR_PATH / "text", exist_ok=True)


def parse_arguments(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)

    # paths
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Path to the data directory"
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="train_data",
        help="Path to the train directory",
    )
    parser.add_argument(
        "--val-dir", type=str, default="val_data", help="Path to the val directory"
    )
    parser.add_argument(
        "--preprocessed-train-dir",
        type=str,
        default="preprocessed_train_data",
        help="Path to the preprocessed train directory",
    )
    parser.add_argument(
        "--preprocessed-val-dir",
        type=str,
        default="preprocessed_val_data",
        help="Path to the preprocessed val directory",
    )

    # meta
    parser.add_argument(
        "--train-csv",
        type=str,
        default="train_data.csv",
        help="Name of the train data CSV file",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default="val_data.csv",
        help="Name of the validation data CSV file",
    )

    # images
    parser.add_argument(
        "--image-size", type=int, default=224, help="Image size for face detection"
    )
    parser.add_argument(
        "--min-face-size", type=int, default=50, help="Minimum face size for detection"
    )
    parser.add_argument("--custom-preprocess", type=bool, default=False)

    # parser.add_argument("", type=str, default="cpu")

    # models
    parser.add_argument("--video-model-device", type=str, default="cuda")
    parser.add_argument("--audio-model-device", type=str, default="cpu")
    parser.add_argument("--text-model-device", type=str, default="cuda")

    # extractions
    parser.add_argument("--extract-video", type=bool, default=False)
    parser.add_argument("--extract-audio", type=bool, default=False)
    parser.add_argument("--extract-text", type=bool, default=False)

    # preprocessing
    parser.add_argument("--preprocess-video", type=bool, default=False)
    parser.add_argument("--preprocess-audio", type=bool, default=False)
    parser.add_argument("--preprocess-text", type=bool, default=False)

    return parser.parse_args()


def get_name(id: int) -> str:
    name = ""
    if id >= 0 and id < 10:
        name = "0000" + str(id)
    elif id >= 10 and id < 100:
        name = "000" + str(id)
    elif id >= 100 and id < 1000:
        name = "00" + str(id)
    elif id >= 1000 and id < 10000:
        name = "0" + str(id)
    else:
        name = str(id)
    return name


# def merge_meta(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
#     DATA_DIR_PATH = Path(args.data_dir)
#     TRAIN_DIR_PATH = DATA_DIR_PATH / args.train_dir
#     VAL_DIR_PATH = DATA_DIR_PATH / args.val_dir
#     PREPROCESSED_TRAIN_DIR_PATH = DATA_DIR_PATH / args.preprocessed_train_dir
#     PREPROCESSED_VAL_DIR_PATH = DATA_DIR_PATH / args.preprocessed_val_dir

#     #############
#     # Meta Data #
#     #############
#     df_train = pd.read_csv(DATA_DIR_PATH / args.train_csv)
#     data = []
#     train_file_paths = sorted(TRAIN_DIR_PATH.glob("*.mp4"))
#     for train_file_path in train_file_paths:
#         _id, q_index, q_type = train_file_path.stem.split("_")
#         data.append(
#             {
#                 "video_id": train_file_path.stem,
#                 "id": _id,
#                 "q_index": q_index,
#                 "q_type": q_type,
#             }
#         )
#     df_train_files = pd.DataFrame(data)
#     df_train_with_meta = pd.merge(df_train_files, df_train, how="left", on="id")
#     df_train_with_meta.to_csv(PREPROCESSED_TRAIN_DIR_PATH / args.train_csv, index=False)

#     df_val = pd.read_csv(DATA_DIR_PATH / args.val_csv)
#     data = []
#     val_file_paths = sorted(VAL_DIR_PATH.glob("*.mp4"))
#     for val_file_path in val_file_paths:
#         _id, q_index, q_type = val_file_path.stem.split("_")
#         data.append(
#             {
#                 "video_id": val_file_path.stem,
#                 "id": _id,
#                 "q_index": q_index,
#                 "q_type": q_type,
#             }
#         )
#     df_val_files = pd.DataFrame(data)
#     df_val_with_meta = pd.merge(df_val_files, df_val, how="left", on="id")
#     df_val_with_meta.to_csv(PREPROCESSED_VAL_DIR_PATH / args.val_csv, index=False)

#     return df_train_with_meta, df_val_with_meta

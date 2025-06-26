import os
import zipfile
import argparse
from pathlib import Path

import pandas as pd

from utils import ensure_paths


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocessing Argument Parser")

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
        "--test-dir", type=str, default="test_data", help="Path to the test directory"
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
    parser.add_argument(
        "--preprocessed-test-dir",
        type=str,
        default="preprocessed_test_data",
        help="Path to the preprocessed test directory",
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
    parser.add_argument(
        "--test-csv",
        type=str,
        default="test_data.csv",
        help="Name of the test data CSV file",
    )

    # images
    parser.add_argument(
        "--image-size", type=int, default=224, help="Image size for face detection"
    )
    parser.add_argument(
        "--min-face-size", type=int, default=50, help="Minimum face size for detection"
    )
    parser.add_argument("--custom-preprocess", type=bool, default=False)

    # models
    parser.add_argument("--video-model-device", type=str, default="cpu")
    parser.add_argument("--audio-model-device", type=str, default="cpu")
    parser.add_argument("--text-model-device", type=str, default="cpu")

    # preprocessing
    parser.add_argument("--preprocess-meta", action="store_true")
    parser.add_argument("--preprocess-video", action="store_true")
    parser.add_argument("--preprocess-audio", action="store_true")
    parser.add_argument("--preprocess-text", action="store_true")
    parser.set_defaults(
        preprocess_meta=False,
        preprocess_video=False,
        preprocess_audio=False,
        preprocess_text=False,
    )

    # parts
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--val", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.set_defaults(train=False, val=False, test=False)

    return parser.parse_args()


def main():
    args = parse_arguments()
    ensure_paths(args.data_dir, args)
    DATA_DIR_PATH = Path(args.data_dir)

    PREPROCESSED_TEST_DIR_PATH = DATA_DIR_PATH / args.preprocessed_test_dir

    df_test = pd.read_csv(PREPROCESSED_TEST_DIR_PATH / args.test_csv)
    personality_labels = [
        "Honesty-Humility",
        "Extraversion",
        "Agreeableness",
        "Conscientiousness",
    ]
    columns = ["id"] + personality_labels

    submit_df = df_test[columns]

    file_path = Path("results/submission.csv")

    os.makedirs("results", exist_ok=True)
    submit_df.to_csv(file_path, index=False)

    with zipfile.ZipFile(
        file_path.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED
    ) as zipf:
        zipf.write(file_path, arcname=os.path.basename(file_path))


if __name__ == "__main__":
    main()

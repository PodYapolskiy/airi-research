import os
import argparse
from pathlib import Path

import cv2
import torch
import mlflow
import numpy as np
import pandas as pd
import moviepy as mp
from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer


def parse_arguments():
    parser = argparse.ArgumentParser(description="Video processing script")

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
    return parser.parse_args()


def _preprocess():
    # TODO: custom preprocessing
    pass


def preprocess(
    df: pd.DataFrame,
    dir_path: Path,
    preprocessed_dir_path: Path,
    mtcnn: MTCNN,
    emoti_eff: EmotiEffLibRecognizer,
):
    for index, row in df.iterrows():
        print(row["id"], row["q_index"], row["q_type"])
        sample_path = dir_path / f"{row['id']}_{row['q_index']}_{row['q_type']}.mp4"

        video = mp.VideoFileClip(sample_path)
        # audio = video.audio  # .write_audiofile(f"{sample_path.stem}.wav")

        cropped_frames: list[np.ndarray] = []
        for t, frame in video.iter_frames(with_times=True):
            cropped_frame: torch.Tensor = mtcnn(frame)

            # prepare for emotiefflib
            # from (c, h, w) to (h, w, c)
            cropped_frame = cropped_frame.permute(1, 2, 0)
            # from [-1, 1] to [0, 255]
            cropped_frame = (cropped_frame + 1) / 2 * 255
            cropped_frame = cropped_frame.clamp(0, 255).to(torch.uint8)

            cropped_frames.append(cropped_frame.numpy())

        # assemble extracted face frames to ensure consisten video content
        if index % 100 == 0:
            # TODO: fix the issue that it si blue (probably somewhere BGR -> RGB)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # MP4V
            frameSize = cropped_frames[0].shape[:2]
            out_path = preprocessed_dir_path / f"{sample_path.stem}_cropped.avi"  # .mp4

            out = cv2.VideoWriter(
                filename=out_path,
                fourcc=fourcc,
                fps=video.fps,
                frameSize=frameSize,
                isColor=True,
            )

            for frame in cropped_frames:
                out.write(frame)

            out.release()
            mlflow.log_artifact(out_path)
            os.remove(out_path)

        features: list[np.ndarray] = []
        batch_size = 32
        for i in range(0, len(cropped_frames), batch_size):
            batch = cropped_frames[i : i + batch_size]
            batch_features = emoti_eff.extract_features(batch)
            features.extend(batch_features)

        # boost performance of conversions list[np.ndarray] -> np.ndarray
        features = np.array(features)
        features_tensor = torch.tensor(features)
        torch.save(
            features_tensor, preprocessed_dir_path / f"{sample_path.stem}_video.pt"
        )
        # print(features_tensor.size())  # [frames, features] (i.e [1680, 1280])

        video.close()
        # audio.close()


def main():
    mlflow.set_tracking_uri(uri="http://localhost:5000")
    mlflow.set_experiment("Preprocessing")

    args = parse_arguments()

    #########
    # PATHS #
    #########
    DATA_DIR_PATH = Path(args.data_dir)

    TRAIN_DIR_PATH = DATA_DIR_PATH / args.train_dir
    VAL_DIR_PATH = DATA_DIR_PATH / args.val_dir

    PREPROCESSED_TRAIN_DIR_PATH = DATA_DIR_PATH / args.preprocessed_train_dir
    PREPROCESSED_VAL_DIR_PATH = DATA_DIR_PATH / args.preprocessed_val_dir

    os.makedirs(TRAIN_DIR_PATH, exist_ok=True)
    os.makedirs(VAL_DIR_PATH, exist_ok=True)
    os.makedirs(PREPROCESSED_TRAIN_DIR_PATH, exist_ok=True)
    os.makedirs(PREPROCESSED_VAL_DIR_PATH, exist_ok=True)

    with mlflow.start_run():
        #############
        # Meta Data #
        #############
        df_train = pd.read_csv(DATA_DIR_PATH / args.train_csv)
        data = []
        train_file_paths = sorted(TRAIN_DIR_PATH.glob("*.mp4"))
        for train_file_path in train_file_paths:
            _id, q_index, q_type = train_file_path.stem.split("_")
            data.append({"id": _id, "q_index": q_index, "q_type": q_type})
        df_train_files = pd.DataFrame(data)
        df_train_with_meta = pd.merge(df_train_files, df_train, how="left", on="id")
        mlflow.log_param("train_size", len(df_train_with_meta))

        df_val = pd.read_csv(DATA_DIR_PATH / args.val_csv)
        data = []
        val_file_paths = sorted(VAL_DIR_PATH.glob("*.mp4"))
        for val_file_path in val_file_paths:
            _id, q_index, q_type = val_file_path.stem.split("_")
            data.append({"id": _id, "q_index": q_index, "q_type": q_type})
        df_val_files = pd.DataFrame(data)
        df_val_with_meta = pd.merge(df_val_files, df_val, how="left", on="id")
        mlflow.log_param("val_size", len(df_val_with_meta))

        ##########
        # Models #
        ##########
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mtcnn = MTCNN(
            image_size=args.image_size, min_face_size=args.min_face_size, device=device
        )
        emoti_eff = EmotiEffLibRecognizer(
            model_name="enet_b0_8_best_vgaf", device=device
        )

        if args.custom_preprocess:
            emoti_eff._preprocess = _preprocess

        #########
        # Train #
        #########
        preprocess(
            df=df_train_with_meta,
            dir_path=TRAIN_DIR_PATH,
            preprocessed_dir_path=PREPROCESSED_TRAIN_DIR_PATH,
            mtcnn=mtcnn,
            emoti_eff=emoti_eff,
        )

        ########
        # Val #
        ########
        preprocess(
            df=df_val_with_meta,
            dir_path=VAL_DIR_PATH,
            preprocessed_dir_path=PREPROCESSED_VAL_DIR_PATH,
            mtcnn=mtcnn,
            emoti_eff=emoti_eff,
        )


if __name__ == "__main__":
    main()

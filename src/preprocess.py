import os
import warnings
import requests
import argparse
from pathlib import Path
from jaxtyping import Float
from rich import print as rprint

import cv2
import torch
import mlflow
import numpy as np
import pandas as pd
import moviepy as mp
from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, EmotiEffLibRecognizerBase
import transformers
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

warnings.filterwarnings("ignore")


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
    emoti_eff: EmotiEffLibRecognizerBase,
    wav2vec2: Wav2Vec2ForCTC,
    wav2vec2_processor: Wav2Vec2Processor,
    device: str = "cpu",
):
    for index, row in df.iterrows():
        sample_path = dir_path / f"{row['video_id']}.mp4"
        rprint(f"{sample_path = }")

        video = mp.VideoFileClip(sample_path)
        audio = video.audio

        #########
        # Video #
        #########
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
        video_features: Float[torch.Tensor, "frames features"] = (  # noqa: F722
            torch.from_numpy(features)
        )
        torch.save(
            obj=video_features,
            f=preprocessed_dir_path / "video" / f"{sample_path.stem}.pt",
        )

        #########
        # Audio #
        #########
        audio_tensor: Float[torch.Tensor, "channels amplitudes"] = (  # noqa: F722
            torch.from_numpy(audio.to_soundarray(fps=audio.fps)).permute(1, 0)
        )
        signal = audio_tensor.mean(dim=0)  # average by channels

        input_values: torch.Tensor = wav2vec2_processor(
            signal, sampling_rate=16000, return_tensors="pt"
        ).input_values

        def hook_fn(module, input, output):
            global hidden_state
            hidden_state = output

        layer = wav2vec2.wav2vec2.encoder.layers[-1]
        hook_handle = layer.register_forward_hook(hook_fn)

        # input_values = input_values.to(device)
        with torch.no_grad():
            _ = wav2vec2(input_values, output_hidden_states=True)

        audio_features = hidden_state[-1].squeeze(0)
        torch.save(
            obj=audio_features,
            f=preprocessed_dir_path / "audio" / f"{sample_path.stem}.pt",
        )

        hook_handle.remove()

        video.close()
        audio.close()

        break


def main():
    MLFLOW_URI = "http://localhost:5000"

    try:
        response = requests.head(MLFLOW_URI)
        response.raise_for_status()
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError):
        raise RuntimeError(
            f"MLflow server is not running at {MLFLOW_URI}. Please start it with `mlflow server`."
        )

    mlflow.set_tracking_uri(uri=MLFLOW_URI)
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
    os.makedirs(PREPROCESSED_TRAIN_DIR_PATH, exist_ok=True)
    os.makedirs(PREPROCESSED_TRAIN_DIR_PATH / "video", exist_ok=True)
    os.makedirs(PREPROCESSED_TRAIN_DIR_PATH / "audio", exist_ok=True)
    os.makedirs(PREPROCESSED_TRAIN_DIR_PATH / "text", exist_ok=True)

    os.makedirs(VAL_DIR_PATH, exist_ok=True)
    os.makedirs(PREPROCESSED_VAL_DIR_PATH, exist_ok=True)
    os.makedirs(PREPROCESSED_VAL_DIR_PATH / "video", exist_ok=True)
    os.makedirs(PREPROCESSED_VAL_DIR_PATH / "audio", exist_ok=True)
    os.makedirs(PREPROCESSED_VAL_DIR_PATH / "text", exist_ok=True)

    with mlflow.start_run():
        #############
        # Meta Data #
        #############
        df_train = pd.read_csv(DATA_DIR_PATH / args.train_csv)
        data = []
        train_file_paths = sorted(TRAIN_DIR_PATH.glob("*.mp4"))
        for train_file_path in train_file_paths:
            _id, q_index, q_type = train_file_path.stem.split("_")
            data.append(
                {
                    "video_id": train_file_path.stem,
                    "id": _id,
                    "q_index": q_index,
                    "q_type": q_type,
                }
            )
        df_train_files = pd.DataFrame(data)
        df_train_with_meta = pd.merge(df_train_files, df_train, how="left", on="id")
        df_train_with_meta.to_csv(
            PREPROCESSED_TRAIN_DIR_PATH / "train_data.csv", index=False
        )
        mlflow.log_param("train_size", len(df_train_with_meta))

        df_val = pd.read_csv(DATA_DIR_PATH / args.val_csv)
        data = []
        val_file_paths = sorted(VAL_DIR_PATH.glob("*.mp4"))
        for val_file_path in val_file_paths:
            _id, q_index, q_type = val_file_path.stem.split("_")
            data.append(
                {
                    "video_id": val_file_path.stem,
                    "id": _id,
                    "q_index": q_index,
                    "q_type": q_type,
                }
            )
        df_val_files = pd.DataFrame(data)
        df_val_with_meta = pd.merge(df_val_files, df_val, how="left", on="id")
        df_val_with_meta.to_csv(PREPROCESSED_VAL_DIR_PATH / "val_data.csv", index=False)
        mlflow.log_param("val_size", len(df_val_with_meta))

        ##########
        # Models #
        ##########
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rprint(f"{device = }")

        mtcnn = MTCNN(
            image_size=args.image_size, min_face_size=args.min_face_size, device=device
        )
        emoti_eff = EmotiEffLibRecognizer(
            model_name="enet_b0_8_best_vgaf", device=device, engine="torch"
        )
        if args.custom_preprocess:
            emoti_eff._preprocess = _preprocess

        model_id = "facebook/wav2vec2-base"
        transformers.logging.set_verbosity_error()
        wav2vec2 = Wav2Vec2ForCTC.from_pretrained(
            model_id, use_safetensors=True
        )  # .to(device)
        wav2vec2.eval()
        wav2vec2_processor = Wav2Vec2Processor.from_pretrained(model_id)

        #########
        # Train #
        #########
        preprocess(
            df=df_train_with_meta,
            dir_path=TRAIN_DIR_PATH,
            preprocessed_dir_path=PREPROCESSED_TRAIN_DIR_PATH,
            mtcnn=mtcnn,
            emoti_eff=emoti_eff,
            wav2vec2=wav2vec2,
            wav2vec2_processor=wav2vec2_processor,
            device=device,
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
            wav2vec2=wav2vec2,
            wav2vec2_processor=wav2vec2_processor,
            device=device,
        )


if __name__ == "__main__":
    main()

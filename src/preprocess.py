import argparse
import warnings
from pathlib import Path
from jaxtyping import Float
import pandas as pd
from rich import print as rprint

from sklearn.preprocessing import MinMaxScaler
import torch
from torch import Tensor
import mlflow
import numpy as np
import moviepy as mp
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, EmotiEffLibRecognizerBase
import torchaudio
import transformers
from transformers import Wav2Vec2ForXVector, Wav2Vec2Processor

from utils import ensure_mlflow, ensure_paths

warnings.filterwarnings("ignore")


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

    # models
    parser.add_argument("--video-model-device", type=str, default="cuda")
    parser.add_argument("--audio-model-device", type=str, default="cpu")
    parser.add_argument("--text-model-device", type=str, default="cuda")

    # preprocessing
    parser.add_argument('--preprocess-meta', action='store_true')
    parser.add_argument('--preprocess-video', action='store_true')
    parser.add_argument('--preprocess-audio', action='store_true')
    parser.add_argument('--preprocess-text', action='store_true')

    parser.set_defaults(
        preprocess_meta=False,
        preprocess_video=False,
        preprocess_audio=False,
        preprocess_text=False
    )

    # parser.add_argument(
    #     "--preprocess-meta",
    #     type=bool,
    #     default=False,
    #     action=argparse.BooleanOptionalAction,
    # )
    # parser.add_argument(
    #     "--preprocess-video",
    #     type=bool,
    #     default=False,
    #     action=argparse.BooleanOptionalAction,
    # )
    # parser.add_argument(
    #     "--preprocess-audio",
    #     type=bool,
    #     default=False,
    #     action=argparse.BooleanOptionalAction,
    # )
    # parser.add_argument(
    #     "--preprocess-text",
    #     type=bool,
    #     default=False,
    #     action=argparse.BooleanOptionalAction,
    # )

    return parser.parse_args()


def preprocess_meta(
    df: pd.DataFrame,
    is_train: bool,
    features_scaler,
):
    categorical_features = ["gender", "education"]
    to_scale_features = ["age", "work_experience"]

    df = pd.get_dummies(df, columns=categorical_features, dtype=int, drop_first=False)

    if is_train:
        features_scaler.fit(df[to_scale_features])

    df[to_scale_features] = features_scaler.transform(df[to_scale_features])

    return df


def preprocess_video(
    preprocessed_dir_path: Path,
    emoti_eff: EmotiEffLibRecognizerBase,
):
    for video_path in sorted(preprocessed_dir_path.glob("video/*.mp4")):
        rprint(f"{video_path = }")

        video = mp.VideoFileClip(video_path)
        cropped_frames: list[np.ndarray] = [frame for frame in video.iter_frames()]
        video.close()

        # apply emotiefflib to extract embedding of each frame's face
        batch_size = 32
        features: list[np.ndarray] = []
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
            f=preprocessed_dir_path / "video" / f"{video_path.stem}.pt",
        )


def preprocess_audio(
    preprocessed_dir_path: Path,
    model: Wav2Vec2ForXVector,
    processor: Wav2Vec2Processor,
    device: torch.device
):
    model.eval()
    for audio_path in sorted(preprocessed_dir_path.glob("audio/*.wav")):
        rprint(f"{audio_path = }")

        waveform, sr = torchaudio.load(audio_path)
        # waveform = torchaudio.functional.vad(waveform, sr)  # trim silence

        signal: Float[Tensor, "channels amplitudes"] = processor(  # noqa: F722 # type: ignore
            waveform, sampling_rate=16000, return_tensors="pt"
        ).input_values.squeeze(0)
        signal = signal.half().to(device)

        with torch.no_grad():
            outputs = model(signal)  # noqa: F722

        audio_features: Float[torch.Tensor, "channels features"] = (  # noqa: F722
            outputs.embeddings.detach().cpu()
        )

        torch.save(
            obj=audio_features,
            f=preprocessed_dir_path / "audio" / f"{audio_path.stem}.pt",
        )


def preprocess_text():
    pass


def main():
    ensure_mlflow()
    mlflow.set_experiment("Preprocessing")
    args = parse_arguments()

    ensure_paths(args.data_dir, args)
    DATA_DIR_PATH = Path(args.data_dir)
    PREPROCESSED_TRAIN_DIR_PATH = DATA_DIR_PATH / args.preprocessed_train_dir
    PREPROCESSED_VAL_DIR_PATH = DATA_DIR_PATH / args.preprocessed_val_dir

    ########
    # Meta #
    ########
    if args.preprocess_meta:
        with mlflow.start_run(run_name="Meta Preprocessing"):
            df_train = pd.read_csv(DATA_DIR_PATH / args.train_csv)
            df_val = pd.read_csv(DATA_DIR_PATH / args.val_csv)

            features_scaler = MinMaxScaler()
            df_train_preprocessed = preprocess_meta(
                df=df_train,
                features_scaler=features_scaler,
                is_train=True,
            )
            df_val_preprocessed = preprocess_meta(
                df=df_val,
                features_scaler=features_scaler,
                is_train=False,
            )

            # to handle missing columns
            left_columns = list(
                set(df_train_preprocessed.columns) - set(df_val_preprocessed.columns)
            )
            df_val_preprocessed[left_columns] = 0

            df_train_preprocessed.to_csv(
                PREPROCESSED_TRAIN_DIR_PATH / args.train_csv, index=False
            )
            df_val_preprocessed.to_csv(
                PREPROCESSED_VAL_DIR_PATH / args.val_csv, index=False
            )

            assert len(df_train_preprocessed.columns) == len(
                df_train_preprocessed.columns
            )
            mlflow.log_param("columns_number", len(df_train_preprocessed.columns))
            mlflow.log_param("train_size", len(df_train_preprocessed))
            mlflow.log_param("val_size", len(df_val_preprocessed))

    #########
    # Video #
    #########
    if args.preprocess_video:
        with mlflow.start_run(run_name="Video Preprocessing"):
            emoti_eff = EmotiEffLibRecognizer(
                model_name="enet_b0_8_best_vgaf",
                device=args.video_model_device,
                engine="torch",
            )

            # TODO: ...
            if args.custom_preprocess:

                def _preprocess():
                    pass

                emoti_eff._preprocess = _preprocess

            preprocess_video(
                preprocessed_dir_path=PREPROCESSED_TRAIN_DIR_PATH, emoti_eff=emoti_eff
            )
            preprocess_video(
                preprocessed_dir_path=PREPROCESSED_VAL_DIR_PATH, emoti_eff=emoti_eff
            )

            emoti_eff.model.to("cpu")
            del emoti_eff

    #########
    # Audio #
    #########
    if args.preprocess_audio:
        with mlflow.start_run(run_name="Audio Preprocessing"):
            transformers.logging.set_verbosity_error()

            model_id = "facebook/wav2vec2-base"
            device = torch.device(args.audio_model_device)
            model = Wav2Vec2ForXVector.from_pretrained(model_id, weights_only=True) # use_safetensors=True, 
            model = model.half().to(device)
            model.eval()
            processor = Wav2Vec2Processor.from_pretrained(model_id)

            preprocess_audio(
                preprocessed_dir_path=PREPROCESSED_TRAIN_DIR_PATH,
                model=model,
                processor=processor,
                device=device,
            )
            preprocess_audio(
                preprocessed_dir_path=PREPROCESSED_VAL_DIR_PATH,
                model=model,
                processor=processor,
                device=device,
            )

            model.to("cpu")
            del model

    ########
    # Text #
    ########
    if args.preprocess_text:
        with mlflow.start_run(run_name="Text Preprocessing"):
            pass


if __name__ == "__main__":
    main()

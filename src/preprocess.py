import os
import argparse
import warnings
from pathlib import Path
import cv2
from rich import print as rprint

import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import torch
from torch import Tensor
from einops import reduce
import torch.nn.functional as F
from jaxtyping import Float
import torchaudio
from tqdm import tqdm
import transformers
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, EmotiEffLibRecognizerBase
from transformers import Wav2Vec2ForXVector, Wav2Vec2Processor
from transformers import AutoModel, AutoTokenizer

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
    video_dir_paths = sorted(
        [path for path in (preprocessed_dir_path / "video").iterdir() if path.is_dir()]
    )

    for video_path in tqdm(
        video_dir_paths,
        desc="Preprocessing Videos",
        total=len(video_dir_paths),
        unit="video",
    ):
        out_path = preprocessed_dir_path / "video" / f"{video_path.stem}.pt"
        if os.path.exists(out_path):
            continue

        video_features = []
        batch_size = 100
        frame_paths = sorted(video_path.glob("*.png"))
        for i in range(0, len(frame_paths), batch_size):
            batch_frame_paths = frame_paths[i : i + batch_size]
            batch_frames: list[np.ndarray] = [
                cv2.imread(str(frame_path), cv2.IMREAD_COLOR_RGB)
                for frame_path in batch_frame_paths
            ]

            batch_video_features = emoti_eff.extract_features(batch_frames)
            video_features.extend(batch_video_features)

        video_features: np.ndarray = np.array(video_features)
        video_features: Float[Tensor, "frames features"] = (  # noqa: F722
            torch.from_numpy(video_features)
        )
        video_features = reduce(video_features, "frames features -> features", "mean")
        assert video_features.shape == (1280,)

        torch.save(obj=video_features, f=out_path)


def preprocess_audio(
    preprocessed_dir_path: Path,
    model: Wav2Vec2ForXVector,
    processor: Wav2Vec2Processor,
    device: torch.device,
):
    model.eval()
    for audio_path in sorted(preprocessed_dir_path.glob("audio/*.wav")):
        rprint(f"{audio_path = }")

        if os.path.exists(audio_path.with_suffix(".pt")):
            continue

        waveform, sr = torchaudio.load(audio_path)
        # waveform = torchaudio.functional.vad(waveform, sr)  # trim silence

        signal: Float[Tensor, "channels amplitudes"] = (
            processor(  # noqa: F722 # type: ignore
                waveform, sampling_rate=16000, return_tensors="pt"
            ).input_values.squeeze(0)
        )
        signal = signal.half().to(device)

        try:
            with torch.no_grad():
                outputs = model(signal)  # noqa: F722
            audio_features: Float[torch.Tensor, "channels features"] = (  # noqa: F722
                outputs.embeddings.detach().cpu()
            )
        except torch.OutOfMemoryError:
            embeddings = []
            chunks = torch.split(signal, 120 * 16000, dim=1)  # 2-mitute chunks
            with torch.no_grad():
                for chunk in chunks:
                    embeddings.append(model(chunk).embeddings)

            audio_features = torch.mean(torch.stack(embeddings), dim=0)

        torch.save(
            obj=audio_features,
            f=preprocessed_dir_path / "audio" / f"{audio_path.stem}.pt",
        )


def preprocess_text(
    preprocessed_dir_path: Path,
    model: AutoModel,
    processor: AutoTokenizer,
):
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    max_length = 8192
    for text_path in sorted(preprocessed_dir_path.glob("text/*.txt")):
        rprint(f"{text_path = }")

        out_path = preprocessed_dir_path / "text" / f"{text_path.stem}.pt"
        if os.path.exists(out_path):
            continue

        with open(text_path, "r") as f:
            text = f.read()

        inputs = processor(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = mean_pooling(outputs, inputs["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        torch.save(
            obj=embeddings.squeeze(0),  # 1024
            f=out_path,
        )


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
            model = Wav2Vec2ForXVector.from_pretrained(
                model_id, weights_only=True
            )  # use_safetensors=True,
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
            # statistics of texts
            for text_path in sorted(PREPROCESSED_TRAIN_DIR_PATH.glob("text/*.txt")):
                with open(text_path, "r") as f:
                    text = f.read()
                    mlflow.log_metric("text_len", len(text))

            model_id = "jinaai/jina-embeddings-v3"  # "Qwen/Qwen3-Embedding-0.6B"
            model = AutoModel.from_pretrained(
                model_id, use_safetensors=True, trust_remote_code=True
            ).to(args.text_model_device)
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

            preprocess_text(
                preprocessed_dir_path=PREPROCESSED_TRAIN_DIR_PATH,
                model=model,
                processor=tokenizer,
            )
            preprocess_text(
                preprocessed_dir_path=PREPROCESSED_VAL_DIR_PATH,
                model=model,
                processor=tokenizer,
            )


if __name__ == "__main__":
    main()

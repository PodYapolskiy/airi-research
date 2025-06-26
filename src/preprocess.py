import os
import cv2
import argparse
import warnings
from tqdm import tqdm
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import torch
from torch import Tensor
from einops import rearrange, reduce

# import torch.nn.functional as F
import torchaudio
import transformers
from jaxtyping import Float
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, EmotiEffLibRecognizerBase

# from transformers import Wav2Vec2ForXVector, Wav2Vec2Processor
from transformers import HubertModel, AutoProcessor
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
    model: HubertModel,
    processor: AutoProcessor,
    device: torch.device,
):
    audio_paths = sorted(preprocessed_dir_path.glob("audio/*.wav"))
    for audio_path in tqdm(
        audio_paths, desc="Preprocessing Audio", total=len(audio_paths), unit="audio"
    ):
        out_path = audio_path.with_suffix(".pt")
        # if os.path.exists(out_path):
        #     continue

        waveform, sr = torchaudio.load(str(audio_path))

        # important to resemple to 16kHz as model expects it
        sampling_rate = 16000
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=sampling_rate
        )
        # waveform = torchaudio.functional.vad(waveform, sr)  # trim silence

        signal: Float[Tensor, "channels amplitudes"] = processor(  # noqa: F722
            audio=waveform, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_values

        signal = rearrange(signal, "1 channels timestemps -> channels timestemps")

        try:
            signal = signal.to(device)
            with torch.no_grad():
                outputs = model(signal)  # noqa: F722

            audio_features = outputs.last_hidden_state.detach().cpu()
            audio_features = reduce(
                audio_features,
                "channels timestemps dim -> channels dim",
                "mean",
            )
            audio_features = reduce(audio_features, "channels dim -> dim", "mean")
        except torch.cuda.OutOfMemoryError:
            embeddings = []
            chunks = torch.split(signal, 60 * 16000, dim=1)  # 1-mitute chunks
            with torch.no_grad():
                for chunk in chunks:
                    chunk = chunk.to(device)
                    outputs = model(chunk)

                    audio_features = outputs.last_hidden_state.detach().cpu()
                    audio_features = reduce(
                        audio_features,
                        "channels timestemps dim -> channels dim",
                        "mean",
                    )

                    embeddings.append(audio_features)

            audio_features = torch.stack(embeddings)
            audio_features = reduce(
                embeddings, "chunks channels dim -> channels dim", "mean"
            )
            audio_features = reduce(audio_features, "channels dim -> dim", "mean")

        torch.save(
            obj=audio_features,
            f=out_path,
        )


def preprocess_text(
    preprocessed_dir_path: Path,
    model: AutoModel,
    processor: AutoTokenizer,
):
    text_paths = sorted(preprocessed_dir_path.glob("text/*.txt"))
    for i, text_path in tqdm(
        enumerate(text_paths),
        desc="Preprocessing Text",
        total=len(text_paths),
        unit="text",
    ):

        out_path = preprocessed_dir_path / "text" / f"{text_path.stem}.pt"
        #     if os.path.exists(out_path):
        #         continue

        with open(text_path, "r") as f:
            text = f.read()

        max_len = 512
        inputs = processor(
            text,
            return_tensors="pt",
            truncation=False,
            # padding=True,
            # max_length=max_len,
        )
        inputs = inputs.to(model.device)
        if inputs.input_ids.size(-1) > max_len:
            # rprint(text_path)
            start_input = inputs.input_ids[:, :max_len]  # torch.clone(
            end_input = inputs.input_ids[:, -max_len:]

            with torch.no_grad():
                start_output = model(start_input)
                end_input = model(end_input)

            assert (
                start_output.last_hidden_state.shape
                == end_input.last_hidden_state.shape
            )

            # [:, 0] is the CLS token for each sequence after output
            start_embedding = start_output.last_hidden_state.detach().cpu()[:, 0]
            end_embedding = end_input.last_hidden_state.detach().cpu()[:, 0]
            embedding = torch.stack([start_embedding, end_embedding])
            embedding = rearrange(embedding, "parts 1 dim -> parts dim")
            embedding = reduce(embedding, "parts dim -> dim", "mean")
        else:  # normal flow
            with torch.no_grad():
                output = model(**inputs)

            embedding = output.last_hidden_state.detach().cpu()[:, 0]
            embedding = rearrange(embedding, "1 dim -> dim")

        assert embedding.shape == (768,)
        torch.save(obj=embedding, f=out_path)


def main():
    ensure_mlflow()
    mlflow.set_experiment("Preprocessing")
    args = parse_arguments()

    ensure_paths(args.data_dir, args)
    DATA_DIR_PATH = Path(args.data_dir)

    PREPROCESSED_TRAIN_DIR_PATH = DATA_DIR_PATH / args.preprocessed_train_dir
    PREPROCESSED_VAL_DIR_PATH = DATA_DIR_PATH / args.preprocessed_val_dir
    PREPROCESSED_TEST_DIR_PATH = DATA_DIR_PATH / args.preprocessed_test_dir

    ########
    # Meta #
    ########
    if args.preprocess_meta:
        with mlflow.start_run(run_name="Meta Preprocessing"):
            df_train = pd.read_csv(DATA_DIR_PATH / args.train_csv)
            df_val = pd.read_csv(DATA_DIR_PATH / args.val_csv)
            df_test = pd.read_csv(DATA_DIR_PATH / args.test_csv)

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
            df_test_preprocessed = preprocess_meta(
                df=df_test,
                features_scaler=features_scaler,
                is_train=False,
            )

            # to handle missing columns
            left_columns = list(
                set(df_train_preprocessed.columns) - set(df_val_preprocessed.columns)
            )
            df_val_preprocessed[left_columns] = 0
            left_columns = list(
                set(df_train_preprocessed.columns) - set(df_test_preprocessed.columns)
            )
            df_test_preprocessed[left_columns] = 0

            drop_columns = list(
                set(df_train_preprocessed.columns) ^ set(df_val_preprocessed.columns)
            )
            df_val_preprocessed.drop(columns=drop_columns, inplace=True)
            drop_columns = list(
                set(df_train_preprocessed.columns) ^ set(df_test_preprocessed.columns)
            )
            df_test_preprocessed.drop(columns=drop_columns, inplace=True)

            df_train_preprocessed.to_csv(
                PREPROCESSED_TRAIN_DIR_PATH / args.train_csv, index=False
            )
            df_val_preprocessed.to_csv(
                PREPROCESSED_VAL_DIR_PATH / args.val_csv, index=False
            )
            df_test_preprocessed.to_csv(
                PREPROCESSED_TEST_DIR_PATH / args.test_csv, index=False
            )

            assert (
                len(df_train_preprocessed.columns)
                == len(df_val_preprocessed.columns)
                == len(df_test_preprocessed.columns)
            ), f"Columns number mismatch {len(df_train_preprocessed.columns)} != {len(df_val_preprocessed.columns)} != {len(df_test_preprocessed.columns)}"

            mlflow.log_param("columns_number", len(df_train_preprocessed.columns))
            mlflow.log_param("train_size", len(df_train_preprocessed))
            mlflow.log_param("val_size", len(df_val_preprocessed))

    #########
    # Video #
    #########
    if args.preprocess_video:
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

        if args.train:
            with mlflow.start_run(run_name="Video Preprocessing (Train)"):
                preprocess_video(
                    preprocessed_dir_path=PREPROCESSED_TRAIN_DIR_PATH,
                    emoti_eff=emoti_eff,
                )

        if args.val:
            with mlflow.start_run(run_name="Video Preprocessing (Val)"):
                preprocess_video(
                    preprocessed_dir_path=PREPROCESSED_VAL_DIR_PATH, emoti_eff=emoti_eff
                )

        if args.test:
            with mlflow.start_run(run_name="Video Preprocessing (Test)"):
                preprocess_video(
                    preprocessed_dir_path=PREPROCESSED_TEST_DIR_PATH,
                    emoti_eff=emoti_eff,
                )

        emoti_eff.model.to("cpu")
        del emoti_eff

    #########
    # Audio #
    #########
    if args.preprocess_audio:
        transformers.logging.set_verbosity_error()

        model_id = "facebook/hubert-xlarge-ls960-ft"
        device = torch.device(args.audio_model_device)
        processor = AutoProcessor.from_pretrained(model_id)
        model = HubertModel.from_pretrained(
            model_id,
            use_safetensors=True,
        ).to(device)
        model.eval()

        if args.train:
            with mlflow.start_run(run_name="Audio Preprocessing (Train)"):
                preprocess_audio(
                    preprocessed_dir_path=PREPROCESSED_TRAIN_DIR_PATH,
                    model=model,
                    processor=processor,
                    device=device,
                )

        if args.val:
            with mlflow.start_run(run_name="Audio Preprocessing (Val)"):
                preprocess_audio(
                    preprocessed_dir_path=PREPROCESSED_VAL_DIR_PATH,
                    model=model,
                    processor=processor,
                    device=device,
                )

        if args.test:
            with mlflow.start_run(run_name="Audio Preprocessing (Test)"):
                preprocess_audio(
                    preprocessed_dir_path=PREPROCESSED_TEST_DIR_PATH,
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
        model_id = "j-hartmann/emotion-english-distilroberta-base"
        model = AutoModel.from_pretrained(model_id, use_safetensors=True).to(
            args.text_model_device
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        if args.train:
            with mlflow.start_run(run_name="Text Preprocessing (Train)"):
                preprocess_text(
                    preprocessed_dir_path=PREPROCESSED_TRAIN_DIR_PATH,
                    model=model,
                    processor=tokenizer,
                )

        if args.val:
            with mlflow.start_run(run_name="Text Preprocessing (Val)"):
                preprocess_text(
                    preprocessed_dir_path=PREPROCESSED_VAL_DIR_PATH,
                    model=model,
                    processor=tokenizer,
                )

        if args.test:
            with mlflow.start_run(run_name="Text Preprocessing (Test)"):
                preprocess_text(
                    preprocessed_dir_path=PREPROCESSED_TEST_DIR_PATH,
                    model=model,
                    processor=tokenizer,
                )

        model.to("cpu")
        del model


if __name__ == "__main__":
    main()

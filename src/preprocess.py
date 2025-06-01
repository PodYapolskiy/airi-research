import warnings
from pathlib import Path
from jaxtyping import Float
from rich import print as rprint

import torch
from torch import Tensor
import mlflow
import numpy as np
import moviepy as mp
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, EmotiEffLibRecognizerBase
import torchaudio
import transformers
from transformers import Wav2Vec2ForXVector, Wav2Vec2Processor

from utils import ensure_mlflow, ensure_paths, merge_meta, parse_arguments

warnings.filterwarnings("ignore")


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
):
    for audio_path in sorted(preprocessed_dir_path.glob("audio/*.wav")):
        rprint(f"{audio_path = }")

        waveform, sr = torchaudio.load(audio_path)
        signal: Float[Tensor, "channels amplitudes"] = processor(  # noqa: F722
            waveform, sampling_rate=16000, return_tensors="pt"
        ).input_values.squeeze(0)

        with torch.no_grad():
            outputs = model(signal)  # noqa: F722

        audio_features: Float[torch.Tensor, "channels features"] = (  # noqa: F722
            outputs.embeddings
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
    args = parse_arguments("Video processing script")

    ensure_paths(args.data_dir, args)
    DATA_DIR_PATH = Path(args.data_dir)
    PREPROCESSED_TRAIN_DIR_PATH = DATA_DIR_PATH / args.preprocessed_train_dir
    PREPROCESSED_VAL_DIR_PATH = DATA_DIR_PATH / args.preprocessed_val_dir

    ########
    # Meta #
    ########
    with mlflow.start_run(run_name="Meta Preprocessing"):
        df_train, df_val = merge_meta(args)
        mlflow.log_param("train_size", len(df_train))
        mlflow.log_param("val_size", len(df_val))

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
            model = Wav2Vec2ForXVector.from_pretrained(model_id, use_safetensors=True)
            model = model.to(args.audio_model_device)
            model.eval()
            processor = Wav2Vec2Processor.from_pretrained(model_id)

            preprocess_audio(
                preprocessed_dir_path=PREPROCESSED_TRAIN_DIR_PATH,
                model=model,
                processor=processor,
            )
            preprocess_audio(
                preprocessed_dir_path=PREPROCESSED_VAL_DIR_PATH,
                model=model,
                processor=processor,
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

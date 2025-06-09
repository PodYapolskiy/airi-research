import os
import subprocess
from typing import List
import cv2
import mlflow
import argparse
import numpy as np
from pathlib import Path
from rich import print as rprint

import torch
from torch import Tensor
from jaxtyping import Float
from facenet_pytorch import MTCNN
import whisper

from utils import ensure_mlflow, ensure_paths

NoneType = type(None)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extraction Argument Parser")

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

    # models
    parser.add_argument("--video-model-device", type=str, default="cpu")
    # parser.add_argument("--audio-model-device", type=str, default="cpu")
    parser.add_argument("--text-model-device", type=str, default="cpu")

    # extractions
    parser.add_argument("--extract-video", action='store_true')
    parser.add_argument("--extract-audio", action='store_true')
    parser.add_argument("--extract-text", action='store_true')

    parser.set_defaults(
        extract_video=False,
        extract_audio=False,
        extract_text=False
    )

    return parser.parse_args()


def extract_video(
    file_paths: List[Path],
    preprocessed_dir_path: Path,
    mtcnn: MTCNN,
    image_size: int,
):
    for file_path in file_paths:
        rprint(f"{file_path = }")

        out_path = preprocessed_dir_path / "video" / f"{file_path.stem}.mp4"
        if os.path.exists(out_path):
            continue

        video = cv2.VideoCapture(str(file_path))
        frames = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frames.append(frame)
        video.release()

        # TODO: not optimal for GPU
        batch_size = 256
        cropped_frames: List[np.ndarray] = []
        for idx in range(0, len(frames), batch_size):
            batch_frames = frames[idx : idx + batch_size]

            #! IMPORTANT
            # in case when face can not be detected on one of the frames in batch
            # MTCNN does not hadle it and output shapes became inconsistent
            # so in case of error process each frame in batch individually
            try:
                batch_cropped_frames: List[Float[Tensor, "channel height width"]] = (
                    mtcnn(batch_frames)
                )

                # if all None handle and return all zeros
                if all(
                    isinstance(cropped_frame, NoneType)
                    for cropped_frame in batch_cropped_frames
                ):
                    batch_cropped_frames = [
                        torch.zeros(3, image_size, image_size)
                    ] * batch_size

            except ValueError:
                batch_cropped_frames = []

                for batch_frame in batch_frames:
                    cropped_frame = mtcnn(batch_frame)

                    if isinstance(cropped_frame, NoneType):
                        cropped_frame = torch.zeros(3, image_size, image_size)

                    batch_cropped_frames.append(cropped_frame)

            # not a batch solution
            # batch_cropped_frames: List[Float[Tensor, "channel height width"]] = []
            # for batch_frame in batch_frames:
            #     cropped_frame = mtcnn(batch_frame)

            #     if isinstance(cropped_frame, NoneType):
            #         cropped_frame = torch.zeros(3, image_size, image_size)

            #     batch_cropped_frames.append(cropped_frame)

            assert all(
                [
                    isinstance(cropped_frame, Tensor)
                    and cropped_frame.shape == (3, image_size, image_size)
                    for cropped_frame in batch_cropped_frames
                ]
            )

            # prepare for emotiefflib
            batch_cropped_frames = [
                cropped_frame.permute(1, 2, 0) for cropped_frame in batch_cropped_frames
            ]
            batch_cropped_frames = [
                (cropped_frame + 1) / 2 * 255  # from [-1, 1] to [0, 255]
                for cropped_frame in batch_cropped_frames
            ]
            batch_cropped_frames = [
                cropped_frame.clamp(0, 255).to(torch.uint8)
                for cropped_frame in batch_cropped_frames
            ]
            batch_cropped_frames = [
                cropped_frame.numpy() for cropped_frame in batch_cropped_frames
            ]

            cropped_frames.extend(batch_cropped_frames)

        # save as video of face frames
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4V
        frameSize = cropped_frames[0].shape[:2]

        out = cv2.VideoWriter(
            filename=out_path,
            fourcc=fourcc,
            fps=video.get(cv2.CAP_PROP_FPS),
            frameSize=frameSize,
            isColor=True,
        )

        for frame in cropped_frames:
            out.write(frame)

        out.release()


def extract_audio(
    file_paths: List[Path],
    preprocessed_dir_path: Path,
):
    for file_path in file_paths:
        rprint(f"{file_path = }")

        out_path = preprocessed_dir_path / "audio" / f"{file_path.stem}.wav"
        if not os.path.exists(out_path):
            rprint(f"Converting {file_path} to {out_path}")
            subprocess.call(
                args=["ffmpeg", "-y", "-i", str(file_path), str(out_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT
            )  


def extract_text(
    file_paths: List[Path], preprocessed_dir_path: Path, model: whisper.Whisper
):
    # check if audio preprocessing has already been accomplished
    audio_files = sorted(preprocessed_dir_path.glob("audio/*.wav"))
    if len(audio_files) != len(file_paths):
        rprint("Extracting audio first...")
        extract_audio(file_paths, preprocessed_dir_path)

    audio_files = sorted(preprocessed_dir_path.glob("audio/*.wav"))
    assert len(audio_files) == len(file_paths)
    file_paths = audio_files

    for file_path in file_paths:
        rprint(f"{file_path = }")

        if os.path.exists(preprocessed_dir_path / "text" / f"{file_path.stem}.txt"):
            continue

        text = model.transcribe(str(file_path), temperature=0)["text"]

        out_path = preprocessed_dir_path / "text" / f"{file_path.stem}.txt"
        with open(out_path, "w") as f:
            f.write(text)


def main():
    ensure_mlflow()
    mlflow.set_experiment("Modalities Extraction")
    args = parse_arguments()

    ensure_paths(args.data_dir, args)
    DATA_DIR_PATH = Path(args.data_dir)
    TRAIN_DIR_PATH = DATA_DIR_PATH / args.train_dir
    VAL_DIR_PATH = DATA_DIR_PATH / args.val_dir
    PREPROCESSED_TRAIN_DIR_PATH = DATA_DIR_PATH / args.preprocessed_train_dir
    PREPROCESSED_VAL_DIR_PATH = DATA_DIR_PATH / args.preprocessed_val_dir

    train_file_paths = sorted(TRAIN_DIR_PATH.glob("*.mp4"))
    val_file_paths = sorted(VAL_DIR_PATH.glob("*.mp4"))

    #########
    # VIDEO #
    #########
    if args.extract_video:
        with mlflow.start_run(run_name="Video Extraction"):
            mlflow.log_param("video_train_size", len(train_file_paths))
            mlflow.log_param("video_val_size", len(val_file_paths))

            mtcnn = MTCNN(
                image_size=args.image_size,
                min_face_size=args.min_face_size,
                device=args.video_model_device,
                selection_method="largest",
            )

            extract_video(
                file_paths=train_file_paths,
                mtcnn=mtcnn,
                image_size=args.image_size,
                preprocessed_dir_path=PREPROCESSED_TRAIN_DIR_PATH,
            )
            extract_video(
                file_paths=val_file_paths,
                mtcnn=mtcnn,
                image_size=args.image_size,
                preprocessed_dir_path=PREPROCESSED_VAL_DIR_PATH,
            )

            mtcnn.to("cpu")
            del mtcnn

    #########
    # AUDIO #
    #########
    if args.extract_audio:
        with mlflow.start_run(run_name="Audio Extraction"):
            mlflow.log_param("audio_train_size", len(train_file_paths))
            mlflow.log_param("audio_val_size", len(val_file_paths))

            extract_audio(
                file_paths=train_file_paths,
                preprocessed_dir_path=PREPROCESSED_TRAIN_DIR_PATH,
            )
            extract_audio(
                file_paths=val_file_paths,
                preprocessed_dir_path=PREPROCESSED_VAL_DIR_PATH,
            )

    ########
    # TEXT #
    ########
    if args.extract_text:
        with mlflow.start_run(run_name="Text Extraction"):
            mlflow.log_param("text_train_size", len(train_file_paths))
            mlflow.log_param("text_val_size", len(val_file_paths))

            model = whisper.load_model(name="small", device=args.text_model_device)

            extract_text(
                file_paths=train_file_paths,
                preprocessed_dir_path=PREPROCESSED_TRAIN_DIR_PATH,
                model=model,
            )
            extract_text(
                file_paths=val_file_paths,
                preprocessed_dir_path=PREPROCESSED_VAL_DIR_PATH,
                model=model,
            )

            model.to("cpu")
            del model


if __name__ == "__main__":
    main()

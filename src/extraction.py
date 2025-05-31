import cv2
import mlflow
import numpy as np
from pathlib import Path
from rich import print as rprint

import torch
from torch import Tensor
import moviepy as mp
from jaxtyping import Float
from facenet_pytorch import MTCNN
import whisper

from utils import ensure_mlflow, ensure_paths, parse_arguments


def extract_video(
    file_paths: list[Path],
    preprocessed_dir_path: Path,
    mtcnn: MTCNN,
):
    for file_path in file_paths:
        rprint(f"{file_path = }")

        video = mp.VideoFileClip(file_path)
        frames = [frame for frame in video.iter_frames()]

        batch_size = 128
        cropped_frames: list[np.ndarray] = []
        for idx in range(0, len(frames), batch_size):
            batch_frames = frames[idx : idx + batch_size]
            batch_cropped_frames: list[Float[Tensor, "channel height width"]] = (  # noqa: F722
                mtcnn(batch_frames)
            )
        
            # prepare for emotiefflib
            batch_cropped_frames = [
                cropped_frame.permute(1, 2, 0)
                for cropped_frame in batch_cropped_frames
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
                cropped_frame.numpy()
                for cropped_frame in batch_cropped_frames
            ]
            
            cropped_frames.extend(batch_cropped_frames)

    
        # save as video of face frames
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4V
        frameSize = cropped_frames[0].shape[:2]
        out_path = preprocessed_dir_path / "video" / f"{file_path.stem}.mp4"

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


def extract_audio(
    file_paths: list[Path],
    preprocessed_dir_path: Path,
):
    for file_path in file_paths:
        rprint(f"{file_path = }")

        video = mp.VideoFileClip(file_path)
        audio = video.audio

        out_path = preprocessed_dir_path / "audio" / f"{file_path.stem}.wav"
        audio.write_audiofile(out_path)


def extract_text(
    file_paths: list[Path],
    preprocessed_dir_path: Path,
    model: whisper.Whisper
):
    # check if audio preprocessing has already been accomplished
    audio_files = sorted(preprocessed_dir_path.glob("audio/*.wav"))
    if len(audio_files) != len(file_paths):
        extract_audio(file_paths, preprocessed_dir_path)

    audio_files = sorted(preprocessed_dir_path.glob("audio/*.wav"))
    assert len(audio_files) == len(file_paths)
    file_paths = audio_files

    for file_path in file_paths:
        rprint(f"{file_path = }")

        text = model.transcribe(str(file_path), temperature=0)["text"]

        out_path = preprocessed_dir_path / "text" / f"{file_path.stem}.txt"
        with open(out_path, "w") as f:
            f.write(text)


def main():
    ensure_mlflow()
    mlflow.set_experiment("Modalities Extraction")
    args = parse_arguments("Modalities extraction.")

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
    with mlflow.start_run(run_name="Video Extraction"):
        mlflow.log_param("video_train_size", len(train_file_paths))
        mlflow.log_param("video_val_size", len(val_file_paths))

        mtcnn = MTCNN(
            image_size=args.image_size,
            min_face_size=args.min_face_size,
            device=args.video_model_device,
        )

        extract_video(
            file_paths=train_file_paths,
            mtcnn=mtcnn,
            preprocessed_dir_path=PREPROCESSED_TRAIN_DIR_PATH,
        )
        extract_video(
            file_paths=val_file_paths,
            mtcnn=mtcnn,
            preprocessed_dir_path=PREPROCESSED_VAL_DIR_PATH,
        )
    
        mtcnn.to("cpu")
        del mtcnn

    #########
    # AUDIO #
    #########
    with mlflow.start_run(run_name="Audio Extraction"):
        mlflow.log_param("audio_train_size", len(train_file_paths))
        mlflow.log_param("audio_val_size", len(val_file_paths))

        extract_audio(
            file_paths=train_file_paths,
            preprocessed_dir_path=PREPROCESSED_TRAIN_DIR_PATH
        )
        extract_audio(
            file_paths=val_file_paths,
            preprocessed_dir_path=PREPROCESSED_VAL_DIR_PATH
        )

    ########
    # TEXT #
    ########
    with mlflow.start_run(run_name="Text Extraction"):
        mlflow.log_param("text_train_size", len(train_file_paths))
        mlflow.log_param("text_val_size", len(val_file_paths))

        model = whisper.load_model(
            name="small",
            device=args.text_model_device
        )
        
        extract_text(
            file_paths=train_file_paths,
            preprocessed_dir_path=PREPROCESSED_TRAIN_DIR_PATH,
            model=model
        )
        extract_text(
            file_paths=val_file_paths,
            preprocessed_dir_path=PREPROCESSED_VAL_DIR_PATH,
            model=model
        )

        model.to("cpu")
        del model


if __name__ == "__main__":
    main()

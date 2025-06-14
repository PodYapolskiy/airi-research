import os
import cv2
import mlflow
import argparse
import subprocess
from tqdm import tqdm
from typing import List
from pathlib import Path
from rich import print as rprint

import torch
import whisper
from facenet_pytorch import MTCNN

from utils import ensure_mlflow, ensure_paths, get_name

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
    parser.add_argument("--extract-video", action="store_true")
    parser.add_argument("--extract-audio", action="store_true")
    parser.add_argument("--extract-text", action="store_true")
    parser.set_defaults(extract_video=False, extract_audio=False, extract_text=False)

    return parser.parse_args()


def extract_video(
    file_paths: List[Path],
    preprocessed_dir_path: Path,
    mtcnn: MTCNN,
    image_size: int,
):
    total_frames = total_duration = total_no_face_videos = 0

    for idx, file_path in tqdm(
        enumerate(file_paths),
        desc="Extracting video",
        total=len(file_paths),
        unit="video",
    ):
        out_path = preprocessed_dir_path / "video" / f"{file_path.stem}"
        os.makedirs(out_path, exist_ok=True)

        video = cv2.VideoCapture(str(file_path))
        fps = video.get(cv2.CAP_PROP_FPS)
        num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = num_frames / fps

        #########
        # Stats #
        #########
        total_frames += num_frames
        total_duration += duration
        mlflow.log_metric("frames", num_frames, step=idx)
        mlflow.log_metric("duration", duration, step=idx)

        no_face = False
        no_faces_frames = 0
        frame_paths = sorted(out_path.glob("*.png"))
        if os.path.exists(out_path):
            for frame_path in frame_paths:
                frame = cv2.imread(str(frame_path))
                if (frame == 0).all():
                    no_face = True
                    no_faces_frames += 1
        else:
            frames = []
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                frames.append(frame)

            # extract face on each frame
            for i, frame in enumerate(frames):
                cropped_frame = mtcnn(frame)
                if isinstance(cropped_frame, NoneType):
                    cropped_frame = torch.zeros(image_size, image_size, 3)
                    cropped_frame = cropped_frame.to(torch.uint8)
                    cropped_frame = cropped_frame.numpy()
                    no_face = True
                    no_faces_frames += 1
                else:
                    cropped_frame = cropped_frame.permute(1, 2, 0)
                    cropped_frame = (cropped_frame + 1) / 2 * 255  # [-1, 1] -> [0, 255]
                    cropped_frame = cropped_frame.clamp(0, 255).to(torch.uint8)
                    cropped_frame = cropped_frame.numpy()

                assert cropped_frame.shape == (image_size, image_size, 3)

                cv2.imwrite(str(out_path / f"{get_name(i)}.png"), cropped_frame)

        if no_face:
            total_no_face_videos += 1

        mlflow.log_metric("no_faces_frames", no_faces_frames, step=idx)

        video.release()

    ###########
    # Metrics #
    ###########
    mlflow.log_metric("total_frames", total_frames)  # total amount of frames
    mlflow.log_metric("total_videos", len(file_paths))  # total amount of videos
    mlflow.log_metric("total_no_face_videos", total_no_face_videos)
    mlflow.log_metric("average_duration", total_duration / len(file_paths))
    mlflow.log_metric("average_frames", total_frames / len(file_paths))


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
                stderr=subprocess.STDOUT,
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
        mtcnn = MTCNN(
            image_size=args.image_size,
            min_face_size=args.min_face_size,
            device=args.video_model_device,
            selection_method="largest",
        )

        with mlflow.start_run(run_name="Video Extraction (Train)"):
            mlflow.log_param("video_train_size", len(train_file_paths))
            extract_video(
                file_paths=train_file_paths,
                mtcnn=mtcnn,
                image_size=args.image_size,
                preprocessed_dir_path=PREPROCESSED_TRAIN_DIR_PATH,
            )

        with mlflow.start_run(run_name="Video Extraction (Val)"):
            mlflow.log_param("video_val_size", len(val_file_paths))
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

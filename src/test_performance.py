import os
import argparse
from pathlib import Path

import mlflow
import pandas as pd
import torch
from torch import Tensor
from jaxtyping import Float
from einops import rearrange

from dataset import Track, get_dataloader
from models import PersonalityNet
from utils import ensure_mlflow, ensure_paths


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training Argument Parser")

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

    # training hyperparameters
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)

    # ...
    parser.add_argument("--trait", type=str, required=True)

    # modalities
    parser.add_argument("--with-meta", action="store_true")
    parser.add_argument("--with-video", action="store_true")
    parser.add_argument("--with-audio", action="store_true")
    parser.add_argument("--with-text", action="store_true")
    parser.set_defaults(
        meta_modality=False,
        video_modality=False,
        audio_modality=False,
        text_modality=False,
    )

    # ...
    parser.add_argument("--meta-dim", type=int, default=13)
    parser.add_argument("--video-dim", type=int, default=1280)
    parser.add_argument("--audio-dim", type=int, default=1280)
    parser.add_argument("--text-dim", type=int, default=768)

    parser.add_argument("--fusion", type=str, default="late", choices=["early", "late"])

    return parser.parse_args()


def main():
    args = parse_arguments()

    assert args.trait in [
        "Honesty-Humility",
        "Extraversion",
        "Agreeableness",
        "Conscientiousness",
    ], f"Invalid personality trait: {args.trait}"

    ensure_mlflow()
    mlflow.set_experiment(f"{args.trait}")

    ensure_paths(args.data_dir, args)
    os.makedirs("models", exist_ok=True)
    MODELS_DIR_PATH = Path("models")
    DATA_DIR_PATH = Path(args.data_dir)

    # PREPROCESSED_TRAIN_DIR_PATH = DATA_DIR_PATH / args.preprocessed_train_dir
    # PREPROCESSED_VAL_DIR_PATH = DATA_DIR_PATH / args.preprocessed_val_dir
    PREPROCESSED_TEST_DIR_PATH = DATA_DIR_PATH / args.preprocessed_test_dir

    run_name = ["Test"]
    # if args.with_meta:
    #     run_name += ["Meta"]
    # if args.with_video:
    #     run_name += ["Video"]
    # if args.with_audio:
    #     run_name += ["Audio"]
    # if args.with_text:
    #     run_name += ["Text"]

    # if args.fusion == "early":
    #     run_name += ["(early fusion)"]
    # elif args.fusion == "late":
    #     run_name += ["(late fusion)"]
    # else:
    #     raise ValueError(f"Invalid fusion strategy: {args.fusion}")
    run_name = " | ".join(run_name)

    with mlflow.start_run(run_name=run_name):
        test_dataloader = get_dataloader(
            track=Track.Personality,
            preprocessed_dir=PREPROCESSED_TEST_DIR_PATH,
            csv=args.test_csv,
            trait=args.trait,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        mlflow.log_params(
            {
                "test_len": len(test_dataloader.dataset),
            }
        )

        device = torch.device(
            args.device
            if torch.cuda.is_available() and args.device.startswith("cuda") == "cuda"
            else "cpu"
        )

        model_path = [
            path
            for path in MODELS_DIR_PATH.glob("*.pt")
            if str(path.stem).startswith(f"[{args.trait}]")
        ]
        if len(model_path) == 0:
            raise FileNotFoundError(f"There is no model for {args.trait} trait!")

        trait, *modalities, fusion = model_path[0].stem.split(" | ")

        assert args.trait in trait
        assert args.with_meta == ("Meta" in modalities)
        assert args.with_video == ("Video" in modalities)
        assert args.with_audio == ("Audio" in modalities)
        assert args.with_text == ("Text" in modalities)
        assert args.fusion in fusion

        model = PersonalityNet(
            with_meta=args.with_meta,
            with_video=args.with_video,
            with_audio=args.with_audio,
            with_text=args.with_text,
            meta_dim=args.meta_dim,
            video_dim=args.video_dim,
            audio_dim=args.audio_dim,
            text_dim=args.text_dim,
            fusion_strategy=args.fusion,
        ).to(device)
        state_dict = torch.load(
            model_path[0],
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(state_dict)
        model.eval()

        answer_df = {
            "video_id": [],
            "answer": [],
        }

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                meta_embeddings: Float[Tensor, "batch dim"] | None = (  # noqa: F722
                    batch["features"].to(device) if model.with_meta else None
                )
                video_embeddings: Float[Tensor, "batch dim"] | None = (  # noqa: F722
                    batch["video_embedding"].to(device) if model.with_video else None
                )
                audio_embeddings: Float[Tensor, "batch dim"] | None = (  # noqa: F722
                    batch["audio_embedding"].to(device) if model.with_audio else None
                )
                text_embeddings: Float[Tensor, "batch dim"] | None = (  # noqa: F722
                    batch["text_embedding"].to(device) if model.with_text else None
                )

                output: Float[Tensor, "batch 1"] = model(  # noqa: F722
                    meta_embeddings, video_embeddings, audio_embeddings, text_embeddings
                )
                output = rearrange(output, "batch 1 -> batch")

                video_ids = batch["video_id"]
                answer_df["video_id"].extend(video_ids)
                answer_df["answer"].extend(output.detach().cpu().tolist())

        answer_df = pd.DataFrame(answer_df)
        os.makedirs("results", exist_ok=True)
        answer_df.to_csv(f"results/{args.trait}.csv", index=False)

        test_df = pd.read_csv(PREPROCESSED_TEST_DIR_PATH / args.test_csv)
        test_df[args.trait] = answer_df["answer"]
        test_df.to_csv(PREPROCESSED_TEST_DIR_PATH / args.test_csv, index=False)


if __name__ == "__main__":
    main()

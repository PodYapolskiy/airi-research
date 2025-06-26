import os
import argparse
from pathlib import Path

import mlflow
import torch
from torch import Tensor
from jaxtyping import Float
from torcheval.metrics import R2Score, MeanSquaredError
from tqdm import tqdm

from models import PersonalityNet
from dataset import get_personality_dataloaders
from utils import ensure_mlflow, ensure_paths


def train_epoch(
    model: PersonalityNet,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.MSELoss,
) -> torch.Tensor:
    model.train()

    total_loss = torch.tensor(0.0)

    for batch_idx, batch in enumerate(train_loader):
        meta_embeddings: Float[Tensor, "batch dim"] | None = (  # noqa: F722
            batch["features"].to(device) if model.with_meta else None
        )
        target: Float[Tensor, "batch"] = batch["targets"].to(device)

        # don't forget to zerograd (don't accumulate grads)
        optimizer.zero_grad()

        # for q in questions:
        video_embeddings: Float[Tensor, "batch dim"] | None = (  # noqa: F722
            batch["video_embedding"].to(device) if model.with_video else None
        )
        audio_embeddings: Float[Tensor, "batch dim"] | None = (  # noqa: F722
            batch["audio_embedding"].to(device) if model.with_audio else None
        )
        text_embeddings: Float[Tensor, "batch dim"] | None = (  # noqa: F722
            batch["text_embedding"].to(device) if model.with_text else None
        )

        # forward pass
        output: Float[Tensor, "batch 1"] = model(  # noqa: F722
            meta_embeddings, video_embeddings, audio_embeddings, text_embeddings
        )
        output: Float[Tensor, "batch"] = output.squeeze(1)

        # backward pass
        loss = criterion(output, target)
        loss.backward()

        total_loss += loss.detach()

        # accumulated by quesions gradients to perform gradient descent step
        optimizer.step()

    return total_loss / len(train_loader)


def eval_epoch(
    model: PersonalityNet,
    device: torch.device,
    val_loader: torch.utils.data.DataLoader,
    criterion,
):
    model.eval()

    total_loss = torch.tensor(0.0)
    mse_metric = MeanSquaredError()
    r2_metric = R2Score()

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            meta_embeddings: Float[Tensor, "batch dim"] | None = (  # noqa: F722
                batch["features"].to(device) if model.with_meta else None
            )
            target: Float[Tensor, "batch"] = batch["targets"].to(device)

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
            output: Float[Tensor, "batch"] = output.squeeze(1)
            loss = criterion(output, target)

            total_loss += loss
            mse_metric.update(output, target)
            r2_metric.update(output, target)

    return total_loss / len(val_loader), mse_metric.compute(), r2_metric.compute()


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
    PREPROCESSED_TRAIN_DIR_PATH = DATA_DIR_PATH / args.preprocessed_train_dir
    PREPROCESSED_VAL_DIR_PATH = DATA_DIR_PATH / args.preprocessed_val_dir

    run_name = []
    if args.with_meta:
        run_name += ["Meta"]
    if args.with_video:
        run_name += ["Video"]
    if args.with_audio:
        run_name += ["Audio"]
    if args.with_text:
        run_name += ["Text"]

    if args.fusion == "early":
        run_name += ["(early fusion)"]
    elif args.fusion == "late":
        run_name += ["(late fusion)"]
    else:
        raise ValueError(f"Invalid fusion strategy: {args.fusion}")
    run_name = " | ".join(run_name)

    with mlflow.start_run(run_name=run_name):
        train_dataloader, val_dataloader = get_personality_dataloaders(
            preprocessed_train_dir=PREPROCESSED_TRAIN_DIR_PATH,
            preprocessed_val_dir=PREPROCESSED_VAL_DIR_PATH,
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            trait=args.trait,
        )
        mlflow.log_params(
            {
                "train_len": len(train_dataloader.dataset),
                "val_len": len(val_dataloader.dataset),
            }
        )

        device = torch.device(
            args.device
            if torch.cuda.is_available() and args.device.startswith("cuda") == "cuda"
            else "cpu"
        )
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        criterion = torch.nn.MSELoss()

        best_val_mse = torch.inf
        for epoch in tqdm(
            range(args.epochs), total=args.epochs, unit="epoch", desc=run_name
        ):
            train_loss = train_epoch(
                model=model,
                device=device,
                train_loader=train_dataloader,
                optimizer=optimizer,
                criterion=criterion,
            )
            mlflow.log_metric("train_loss", train_loss, epoch)

            val_loss, val_mse, val_r2 = eval_epoch(
                model=model,
                device=device,
                val_loader=val_dataloader,
                criterion=criterion,
            )
            mlflow.log_metric("val_loss", val_loss, epoch)
            mlflow.log_metric("val_mse", val_mse, epoch)
            mlflow.log_metric("val_r2", val_r2, epoch)

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                torch.save(
                    model.state_dict(),
                    MODELS_DIR_PATH / f"[{args.trait}] | {run_name}.pt",
                )


if __name__ == "__main__":
    main()

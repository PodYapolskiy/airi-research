import argparse
from pathlib import Path
from rich import print as rprint

import mlflow
import torch
from torch import Tensor
from jaxtyping import Float
from torcheval.metrics import R2Score, MeanSquaredError

from models import PerformanceNet
from dataset import get_performance_dataloaders
from utils import ensure_mlflow, ensure_paths

questions = [f"q{i}" for i in range(1, 7)]


def train_epoch(
    model: PerformanceNet,
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

        for q in questions:
            video_embeddings: Float[Tensor, "batch dim"] | None = (  # noqa: F722
                batch[q]["video_embedding"].to(device) if model.with_video else None
            )
            audio_embeddings: Float[Tensor, "batch dim"] | None = (  # noqa: F722
                batch[q]["audio_embedding"].to(device) if model.with_audio else None
            )
            text_embeddings: Float[Tensor, "batch dim"] | None = (  # noqa: F722
                batch[q]["text_embedding"].to(device) if model.with_text else None
            )

            # forward pass
            output: Float[Tensor, "batch 1"] = model(  # noqa: F722
                meta_embeddings, video_embeddings, audio_embeddings, text_embeddings
            )
            output: Float[Tensor, "batch"] = output.squeeze(1)

            # backward pass
            loss = criterion(output, target)
            # loss = loss / len(questions)
            loss.backward()

            total_loss += loss.detach() / len(questions)

        # accumulated by quesions gradients to perform gradient descent step
        optimizer.step()

    return total_loss / len(train_loader)


def eval_epoch(
    model: PerformanceNet,
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

            for q in questions:
                video_embeddings: Float[Tensor, "batch dim"] | None = (  # noqa: F722
                    batch[q]["video_embedding"].to(device) if model.with_video else None
                )
                audio_embeddings: Float[Tensor, "batch dim"] | None = (  # noqa: F722
                    batch[q]["audio_embedding"].to(device) if model.with_audio else None
                )
                text_embeddings: Float[Tensor, "batch dim"] | None = (  # noqa: F722
                    batch[q]["text_embedding"].to(device) if model.with_text else None
                )

                output: Float[Tensor, "batch 1"] = model(  # noqa: F722
                    meta_embeddings, video_embeddings, audio_embeddings, text_embeddings
                )
                output: Float[Tensor, "batch"] = output.squeeze(1)
                loss = criterion(output, target)
                # loss = loss / len(questions)

                total_loss += loss / len(questions)
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

    # training hyperparameters
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)

    # ...
    parser.add_argument("--trait", type=str, required=True)
    # parser.add_argument("--only-dim", type=int, default=1280)

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
    parser.add_argument("--audio-dim", type=int, default=512)
    parser.add_argument("--text-dim", type=int, default=1024)

    parser.add_argument("--fusion", type=str, default="late", choices=["early", "late"])

    return parser.parse_args()


def main():
    args = parse_arguments()

    assert args.trait in [
        "Integrity",
        "Collegiality",
        "Social_versatility",
        "Development_orientation",
        "Hireability",
    ], f"Invalid performance trait: {args.trait}"

    ensure_mlflow()
    mlflow.set_experiment(f"{args.trait}")

    ensure_paths(args.data_dir, args)
    DATA_DIR_PATH = Path(args.data_dir)
    PREPROCESSED_TRAIN_DIR_PATH = DATA_DIR_PATH / args.preprocessed_train_dir
    PREPROCESSED_VAL_DIR_PATH = DATA_DIR_PATH / args.preprocessed_val_dir

    # with mlflow.start_run(run_name="OnlyNet Baseline"):
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
        train_dataloader, val_dataloader = get_performance_dataloaders(
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
        model = PerformanceNet(
            with_meta=args.with_meta,
            with_video=args.with_video,
            with_audio=args.with_audio,
            with_text=args.with_text,
            meta_dim=args.meta_dim,
            video_dim=args.video_dim,
            audio_dim=args.audio_dim,
            text_dim=args.text_dim,
            fusion_strategy="late",
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        criterion = torch.nn.MSELoss()

        for epoch in range(args.epochs):
            rprint("Epoch: ", epoch)
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


if __name__ == "__main__":
    main()

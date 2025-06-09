import argparse
from pathlib import Path
from rich import print as rprint

import mlflow
import torch
from torch import Tensor
from jaxtyping import Float
from torcheval.metrics import R2Score, MeanSquaredError

from models import OnlyNet
from dataset import get_personality_dataloaders
from utils import ensure_mlflow, ensure_paths


def train_epoch(
    model: torch.nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.MSELoss,
) -> torch.Tensor:
    model.train()

    total_loss = torch.tensor(0.0)

    for batch_idx, batch in enumerate(train_loader):
        audio_embeddings: Float[Tensor, "batch audio_dim"] = batch[  # noqa: F722
            "audio_embedding"
        ].to(device)
        target: Float[Tensor, "batch"] = batch["targets"].to(device)

        # don't forget to zerograd (don't accumulate grads)
        optimizer.zero_grad()

        # forward pass
        output: Float[Tensor, "batch 1"] = model(audio_embeddings)  # noqa: F722
        output: Float[Tensor, "batch"] = output.squeeze(1)

        # backward pass
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        total_loss += loss  # .item()

    return total_loss / len(train_loader)


def eval_epoch(
    model: torch.nn.Module,
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
            audio_embeddings: Float[Tensor, "batch audio_dim"] = batch[  # noqa: F722
                "audio_embedding"
            ].to(device)
            target: Float[Tensor, "batch"] = batch["targets"].to(device)

            output: Float[Tensor, "batch 1"] = model(audio_embeddings)  # noqa: F722
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
    parser.add_argument("--trait", type=str, default="Honesty-Humility")
    parser.add_argument("--only-dim", type=int, default=1280)

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
    mlflow.set_experiment(f"Training Personality {args.trait}")
    ensure_paths(args.data_dir, args)
    DATA_DIR_PATH = Path(args.data_dir)
    PREPROCESSED_TRAIN_DIR_PATH = DATA_DIR_PATH / args.preprocessed_train_dir
    PREPROCESSED_VAL_DIR_PATH = DATA_DIR_PATH / args.preprocessed_val_dir

    train_dataloader, val_dataloader = get_personality_dataloaders(
        preprocessed_train_dir=PREPROCESSED_TRAIN_DIR_PATH,
        preprocessed_val_dir=PREPROCESSED_VAL_DIR_PATH,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        batch_size=args.batch_size,
        num_workers=1, # args.num_workers,
        personality_trait=args.trait,
    )
    mlflow.log_params(
        {
            "train_len": len(train_dataloader.dataset),
            "val_len": len(val_dataloader.dataset),
        }
    )

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.device.startswith("cuda") == "cuda" else "cpu"
    )
    model = OnlyNet(dim=args.only_dim).to(device)
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

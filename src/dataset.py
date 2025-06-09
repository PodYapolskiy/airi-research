import os
import pandas as pd
from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

# from jaxtyping import Float


def get_tensor_from_path(path: str, dim: int) -> Tensor:
    if os.path.exists(path):
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        # Average across frames if needed
        if len(tensor.shape) > 1:
            tensor = torch.mean(tensor, dim=0)
    else:
        tensor = torch.zeros(dim)

    assert tensor.size(-1) == dim
    return tensor


class PersonalityDataset(Dataset):
    """
    Dataset class for multimodal personality traits prediction.
    Combines video, audio, and text embeddings with demographic features.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        preprocessed_dir_path: Path,
        personality_trait: str,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            df (pd.DataFrame): DataFrame with metadata and labels
            preprocessed_dir_path (Path): Path to the directory containing embeddings
            personality_trait (str): The personality trait to predict
        """
        personality_labels = [
            "Honesty-Humility",
            "Extraversion",
            "Agreeableness",
            "Conscientiousness",
        ]
        assert (
            personality_trait in personality_labels
        ), f"Invalid personality trait: {personality_trait}"

        self.video_paths: list[str] = []
        self.audio_paths: list[str] = []
        self.text_paths: list[str] = []
        self.features: list[list[float]] = []
        self.labels: list[float] = []

        features_columns = [
            col
            for col in df.columns
            if col in ["age", "work_experience"]
            or col.startswith("gender_")
            or col.startswith("education_")
        ]

        for _, row in df.iterrows():
            postfix = f"_q{personality_labels.index(personality_trait)+3}_personality"

            video_path = preprocessed_dir_path / "video" / f"{row['id']}{postfix}.pt"
            audio_path = preprocessed_dir_path / "audio" / f"{row['id']}{postfix}.pt"
            text_path = preprocessed_dir_path / "text" / f"{row['id']}{postfix}.pt"

            self.video_paths.append(str(video_path))
            self.audio_paths.append(str(audio_path))
            self.text_paths.append(str(text_path))
            self.features.append(row[features_columns].to_list())
            self.labels.append(row[personality_trait])

        # Make sure that all modalities and feature lists have the same length
        assert (
            len(self.video_paths)
            == len(self.audio_paths)
            == len(self.text_paths)
            == len(self.features)
            == len(self.labels)
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        video_tensor = get_tensor_from_path(self.video_paths[idx], 1280)
        audio_tensor = get_tensor_from_path(self.audio_paths[idx], 512).float()
        text_tensor = get_tensor_from_path(self.text_paths[idx], 1024)

        return {
            "video_embedding": video_tensor,
            "audio_embedding": audio_tensor,
            "text_embedding": text_tensor,
            "features": torch.tensor(self.features[idx]),
            "targets": torch.tensor(self.labels[idx]),
        }


class PerformanceDataset(Dataset):
    """
    Dataset class for multimodal personality traits prediction.
    Combines video, audio, and text embeddings with demographic features.
    """

    def __init__(self, df: pd.DataFrame, preprocessed_dir_path: Path) -> None:
        """
        Initialize the dataset.

        Args:
            df (pd.DataFrame): DataFrame with metadata and labels
            preprocessed_dir_path (Path): Path to the directory containing embeddings
        """
        self.video_paths: list[list[str]] = []
        self.audio_paths: list[list[str]] = []
        self.text_paths: list[list[str]] = []
        self.features: list[list[float]] = []
        self.labels: list[list[float]] = []

        # Define target columns - focusing on the 5 traits mentioned in the task
        target_columns = [
            "Integrity",
            "Collegiality",
            "Social_versatility",
            "Development_orientation",
            "Hireability",
        ]
        postfixes = [
            "_q1_generic",
            "_q2_generic",
            "_q3_personality",
            "_q4_personality",
            "_q5_personality",
            "_q6_personality",
        ]

        for _, row in df.iterrows():
            video_paths = []
            audio_paths = []
            text_paths = []

            for postfix in postfixes:
                video_path = (
                    preprocessed_dir_path / "video" / f"{row['id']}{postfix}.pt"
                )
                audio_path = (
                    preprocessed_dir_path / "audio" / f"{row['id']}{postfix}.pt"
                )
                text_path = preprocessed_dir_path / "text" / f"{row['id']}{postfix}.pt"

                video_paths.append(str(video_path))
                audio_paths.append(str(audio_path))
                text_paths.append(str(text_path))

            features_columns = [
                col
                for col in df.columns
                if col in ["age", "work_experience"]
                or col.startswith("gender_")
                or col.startswith("education_")
            ]

            self.video_paths.append(video_paths)
            self.audio_paths.append(audio_paths)
            self.text_paths.append(text_paths)
            self.features.append(row[features_columns].to_list())
            self.labels.append(row[target_columns].to_list())

        assert (
            len(self.video_paths)
            == len(self.audio_paths)
            == len(self.text_paths)
            == len(self.features)
            == len(self.labels)
        )

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx) -> dict:
        video_paths = self.video_paths[idx]
        audio_paths = self.audio_paths[idx]
        text_paths = self.text_paths[idx]

        q = 1
        interview_answers = {}
        for video_path, audio_path, text_path in zip(
            video_paths, audio_paths, text_paths
        ):
            video_tensor = get_tensor_from_path(video_path, 1280)
            audio_tensor = get_tensor_from_path(audio_path, 512)
            text_tensor = get_tensor_from_path(text_path, 1024)

            interview_answers[f"q{q}"] = {
                "video_embedding": video_tensor,
                "audio_embedding": audio_tensor,
                "text_embedding": text_tensor,
            }
            q += 1

        interview_answers["targets"] = Tensor(self.labels[idx])
        interview_answers["features"] = Tensor(self.features[idx])

        return interview_answers


def get_personality_dataloaders(
    preprocessed_train_dir: Path,
    preprocessed_val_dir: Path,
    train_csv: str,
    val_csv: str,
    personality_trait: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        train_df (pd.DataFrame): Training DataFrame
        val_df (pd.DataFrame): Validation DataFrame
        data_dir (str): Path to the data directory
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for dataloaders

    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    df_train = pd.read_csv(preprocessed_train_dir / train_csv)
    df_val = pd.read_csv(preprocessed_val_dir / val_csv)

    train_dataset = PersonalityDataset(
        df=df_train,
        preprocessed_dir_path=preprocessed_train_dir,
        personality_trait=personality_trait,
    )
    val_dataset = PersonalityDataset(
        df=df_val,
        preprocessed_dir_path=preprocessed_val_dir,
        personality_trait=personality_trait,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        # pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        # pin_memory=True,
    )

    return train_dataloader, val_dataloader

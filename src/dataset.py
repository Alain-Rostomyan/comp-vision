"""
Dataset classes for loading emoji images.
"""
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from src.config import (
    TRAIN_DIR, TEST_DIR, TRAIN_LABELS_PATH,
    LABEL_TO_IDX, IDX_TO_LABEL, SEED, VAL_SPLIT,
    BATCH_SIZE, NUM_WORKERS
)


class EmojiDataset(Dataset):
    """Dataset class for emoji images."""

    def __init__(
        self,
        image_ids: List[str],
        labels: Optional[List[int]] = None,
        image_dir: Path = TRAIN_DIR,
        transform: Optional[Callable] = None
    ):
        """
        Args:
            image_ids: List of image IDs (without extension)
            labels: List of integer labels (None for test set)
            image_dir: Directory containing images
            transform: Transformations to apply to images
        """
        self.image_ids = image_ids
        self.labels = labels
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_id = self.image_ids[idx]
        img_path = self.image_dir / f"{img_id}.png"

        # Load image and convert to RGB (via RGBA to handle palette transparency)
        image = Image.open(img_path)
        if image.mode in ("P", "PA"):
            image = image.convert("RGBA").convert("RGB")
        else:
            image = image.convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Return image and label (or -1 for test set)
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image, -1


def load_train_data() -> pd.DataFrame:
    """Load training labels from CSV."""
    df = pd.read_csv(TRAIN_LABELS_PATH)
    # Convert string labels to integers
    df['label_idx'] = df['Label'].map(LABEL_TO_IDX)
    # Ensure ID is string with leading zeros
    df['Id'] = df['Id'].astype(str).str.zfill(5)
    return df


def get_train_val_split(
    df: pd.DataFrame,
    val_split: float = VAL_SPLIT,
    seed: int = SEED,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and validation sets.
    Uses stratified split to maintain class distribution.
    """
    stratify_col = df['label_idx'] if stratify else None

    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        random_state=seed,
        stratify=stratify_col
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def get_test_ids() -> List[str]:
    """Get list of test image IDs."""
    test_files = sorted(TEST_DIR.glob("*.png"))
    return [f.stem for f in test_files]


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_transform: Callable,
    val_transform: Callable,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""

    train_dataset = EmojiDataset(
        image_ids=train_df['Id'].tolist(),
        labels=train_df['label_idx'].tolist(),
        image_dir=TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = EmojiDataset(
        image_ids=val_df['Id'].tolist(),
        labels=val_df['label_idx'].tolist(),
        image_dir=TRAIN_DIR,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def create_test_dataloader(
    test_transform: Callable,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS
) -> Tuple[DataLoader, List[str]]:
    """Create test dataloader and return IDs."""

    test_ids = get_test_ids()

    test_dataset = EmojiDataset(
        image_ids=test_ids,
        labels=None,
        image_dir=TEST_DIR,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return test_loader, test_ids


def get_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """
    Calculate class weights for handling imbalanced data.
    Uses inverse frequency weighting.
    """
    class_counts = df['label_idx'].value_counts().sort_index()
    total = len(df)
    weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights.values)

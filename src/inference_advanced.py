"""
Advanced inference utilities with Test-Time Augmentation (TTA) and ensemble support.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast
from tqdm import tqdm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import (
    DEVICE, MODEL_DIR, SUBMISSION_DIR, TEST_DIR,
    IDX_TO_LABEL, NUM_CLASSES
)
from src.transforms_advanced import IMAGENET_MEAN, IMAGENET_STD


class TTADataset(Dataset):
    """Dataset that applies TTA transforms to images."""

    def __init__(
        self,
        image_paths: List[Path],
        transforms_list: List[A.Compose],
        image_ids: Optional[List[str]] = None
    ):
        self.image_paths = image_paths
        self.transforms_list = transforms_list
        self.image_ids = image_ids or [p.stem for p in image_paths]

    def __len__(self):
        return len(self.image_paths) * len(self.transforms_list)

    def __getitem__(self, idx):
        img_idx = idx // len(self.transforms_list)
        transform_idx = idx % len(self.transforms_list)

        image_path = self.image_paths[img_idx]
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        transform = self.transforms_list[transform_idx]
        augmented = transform(image=image)

        return augmented['image'], img_idx, transform_idx


def get_tta_transforms(img_size: int = 224, num_tta: int = 5) -> List[A.Compose]:
    """
    Get TTA transforms.

    Args:
        img_size: Image size
        num_tta: Number of TTA augmentations (1=no TTA, 5=standard, 10=heavy)

    Returns:
        List of transforms to apply
    """
    transforms = []

    # Base transform (always included)
    transforms.append(A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ]))

    if num_tta >= 2:
        # Horizontal flip
        transforms.append(A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ]))

    if num_tta >= 4:
        # Rotations
        for angle in [-10, 10]:
            transforms.append(A.Compose([
                A.Resize(img_size, img_size),
                A.Rotate(limit=(angle, angle), border_mode=0, p=1.0),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2()
            ]))

    if num_tta >= 6:
        # Scale variations
        for scale in [0.9, 1.1]:
            size = int(img_size * scale)
            transforms.append(A.Compose([
                A.Resize(size, size),
                A.CenterCrop(img_size, img_size) if scale > 1 else A.PadIfNeeded(img_size, img_size, border_mode=0),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2()
            ]))

    if num_tta >= 8:
        # Brightness variations
        for brightness in [-0.1, 0.1]:
            transforms.append(A.Compose([
                A.Resize(img_size, img_size),
                A.RandomBrightnessContrast(brightness_limit=(brightness, brightness), contrast_limit=0, p=1.0),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2()
            ]))

    if num_tta >= 10:
        # Combined augmentations
        transforms.append(A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=1.0),
            A.Rotate(limit=(-5, -5), border_mode=0, p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ]))
        transforms.append(A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=1.0),
            A.Rotate(limit=(5, 5), border_mode=0, p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ]))

    return transforms[:num_tta]


@torch.no_grad()
def predict_with_tta(
    model: nn.Module,
    image_paths: List[Path],
    img_size: int = 224,
    num_tta: int = 5,
    batch_size: int = 32,
    device: torch.device = DEVICE,
    use_amp: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions with Test-Time Augmentation.

    Args:
        model: Trained model
        image_paths: List of image paths
        img_size: Image size for the model
        num_tta: Number of TTA augmentations
        batch_size: Batch size for inference
        device: Device to use
        use_amp: Use automatic mixed precision

    Returns:
        predictions: Final predicted classes
        probabilities: Averaged probabilities
    """
    model.eval()
    model.to(device)

    transforms_list = get_tta_transforms(img_size, num_tta)
    num_images = len(image_paths)
    num_transforms = len(transforms_list)

    # Store probabilities for each image across all TTA transforms
    all_probs = np.zeros((num_images, NUM_CLASSES))
    tta_counts = np.zeros(num_images)

    print(f"Running TTA with {num_transforms} augmentations on {num_images} images...")

    for t_idx, transform in enumerate(tqdm(transforms_list, desc="TTA transforms")):
        # Process each transform
        for i in range(0, num_images, batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []

            for path in batch_paths:
                image = Image.open(path).convert("RGB")
                image = np.array(image)
                augmented = transform(image=image)
                batch_images.append(augmented['image'])

            batch_tensor = torch.stack(batch_images).to(device)

            with autocast(enabled=use_amp):
                outputs = model(batch_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()

            all_probs[i:i+len(batch_paths)] += probs
            tta_counts[i:i+len(batch_paths)] += 1

    # Average probabilities
    all_probs /= tta_counts[:, np.newaxis]
    predictions = np.argmax(all_probs, axis=1)

    return predictions, all_probs


@torch.no_grad()
def predict_ensemble_with_tta(
    models: List[nn.Module],
    image_paths: List[Path],
    img_sizes: List[int],
    num_tta: int = 5,
    batch_size: int = 32,
    device: torch.device = DEVICE,
    model_weights: Optional[List[float]] = None,
    use_amp: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions with multiple models and TTA.

    Args:
        models: List of trained models
        image_paths: List of image paths
        img_sizes: Image size for each model
        num_tta: Number of TTA augmentations
        batch_size: Batch size for inference
        device: Device to use
        model_weights: Weights for each model (default: equal weights)
        use_amp: Use automatic mixed precision

    Returns:
        predictions: Final predicted classes
        probabilities: Weighted average probabilities
    """
    num_models = len(models)
    if model_weights is None:
        model_weights = [1.0 / num_models] * num_models

    num_images = len(image_paths)
    ensemble_probs = np.zeros((num_images, NUM_CLASSES))

    for model_idx, (model, img_size, weight) in enumerate(zip(models, img_sizes, model_weights)):
        print(f"\nModel {model_idx + 1}/{num_models} (weight: {weight:.2f}):")
        _, probs = predict_with_tta(
            model, image_paths, img_size, num_tta, batch_size, device, use_amp
        )
        ensemble_probs += weight * probs

    predictions = np.argmax(ensemble_probs, axis=1)
    return predictions, ensemble_probs


@torch.no_grad()
def predict_simple(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = DEVICE,
    use_amp: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple prediction without TTA."""
    model.eval()
    model.to(device)

    all_preds = []
    all_probs = []

    for images, _ in tqdm(dataloader, desc="Predicting"):
        images = images.to(device)

        with autocast(enabled=use_amp):
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_probs)


def create_submission(
    predictions: np.ndarray,
    test_ids: List[str],
    filename: Optional[str] = None,
    use_labels: bool = True
) -> pd.DataFrame:
    """Create submission CSV file."""
    if use_labels:
        pred_labels = [IDX_TO_LABEL[p] for p in predictions]
    else:
        pred_labels = predictions.tolist()

    submission = pd.DataFrame({
        'ID': test_ids,
        'Label': pred_labels
    })

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"submission_{timestamp}.csv"

    output_path = SUBMISSION_DIR / filename
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")

    return submission


def load_model_for_inference(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device = DEVICE
) -> nn.Module:
    """Load model weights for inference."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    if 'best_val_acc' in checkpoint:
        print(f"  Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
    return model


def analyze_predictions(
    probabilities: np.ndarray,
    predictions: np.ndarray,
    threshold: float = 0.7
) -> Dict:
    """Analyze prediction confidence."""
    max_probs = probabilities.max(axis=1)

    analysis = {
        'total_samples': len(predictions),
        'mean_confidence': max_probs.mean(),
        'min_confidence': max_probs.min(),
        'max_confidence': max_probs.max(),
        'low_confidence_count': (max_probs < threshold).sum(),
        'high_confidence_count': (max_probs >= 0.9).sum(),
        'class_distribution': {},
    }

    for cls_idx in range(NUM_CLASSES):
        count = (predictions == cls_idx).sum()
        analysis['class_distribution'][IDX_TO_LABEL[cls_idx]] = count

    return analysis

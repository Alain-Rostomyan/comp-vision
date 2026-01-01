"""
Inference utilities and submission generation.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from src.config import (
    DEVICE, MODEL_DIR, SUBMISSION_DIR, TEST_DIR,
    IDX_TO_LABEL, LABEL_TO_IDX, NUM_CLASSES
)
from src.dataset import create_test_dataloader, EmojiDataset


@torch.no_grad()
def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = DEVICE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions on a dataloader.

    Returns:
        predictions: Array of predicted class indices
        probabilities: Array of prediction probabilities (softmax)
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_probs = []

    for images, _ in tqdm(dataloader, desc="Predicting"):
        images = images.to(device)
        outputs = model(images)

        # Get probabilities and predictions
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_probs)


def create_submission(
    predictions: np.ndarray,
    test_ids: List[str],
    filename: Optional[str] = None,
    use_labels: bool = False
) -> pd.DataFrame:
    """
    Create submission CSV file.

    Args:
        predictions: Array of predicted class indices
        test_ids: List of test image IDs
        filename: Output filename (auto-generated if None)
        use_labels: If True, use string labels; if False, use indices

    Returns:
        DataFrame with predictions
    """
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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def get_misclassified_examples(
    model: nn.Module,
    dataloader: DataLoader,
    image_ids: List[str],
    labels: List[int],
    num_examples: int = 10,
    device: torch.device = DEVICE
) -> List[dict]:
    """
    Find misclassified examples for analysis.

    Returns list of dicts with:
        - image_id
        - true_label
        - predicted_label
        - confidence
    """
    model.eval()
    model.to(device)

    misclassified = []

    idx = 0
    for images, true_labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        confidences, preds = probs.max(1)

        for i in range(len(images)):
            if preds[i].item() != true_labels[i].item():
                misclassified.append({
                    'image_id': image_ids[idx + i],
                    'true_label': IDX_TO_LABEL[true_labels[i].item()],
                    'predicted_label': IDX_TO_LABEL[preds[i].item()],
                    'confidence': confidences[i].item(),
                    'true_idx': true_labels[i].item(),
                    'pred_idx': preds[i].item()
                })

                if len(misclassified) >= num_examples:
                    return misclassified

        idx += len(images)

    return misclassified


def predict_single_image(
    model: nn.Module,
    image_path: Path,
    transform,
    device: torch.device = DEVICE
) -> Tuple[str, float, np.ndarray]:
    """
    Predict label for a single image.

    Returns:
        predicted_label: String label
        confidence: Prediction confidence
        all_probs: Probabilities for all classes
    """
    model.eval()
    model.to(device)

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = probs.max(1)

    return (
        IDX_TO_LABEL[pred.item()],
        confidence.item(),
        probs.cpu().numpy()[0]
    )


def ensemble_predictions(
    models: List[nn.Module],
    dataloader: DataLoader,
    device: torch.device = DEVICE,
    method: str = "soft"
) -> np.ndarray:
    """
    Ensemble predictions from multiple models.

    Args:
        models: List of trained models
        dataloader: Test dataloader
        device: Device to use
        method: 'soft' for probability averaging, 'hard' for voting

    Returns:
        Final predictions
    """
    all_probs = []

    for model in models:
        _, probs = predict(model, dataloader, device)
        all_probs.append(probs)

    if method == "soft":
        # Average probabilities
        avg_probs = np.mean(all_probs, axis=0)
        predictions = np.argmax(avg_probs, axis=1)
    else:  # hard voting
        all_preds = [np.argmax(p, axis=1) for p in all_probs]
        predictions = np.array([
            np.bincount(votes).argmax()
            for votes in zip(*all_preds)
        ])

    return predictions

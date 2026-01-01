"""
Configuration settings for the emoji classification project.
"""
import os
from pathlib import Path

# Random seed for reproducibility
SEED = 42

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "2-computer-vision-2025-b-sc-aidams-final-proj"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
TRAIN_LABELS_PATH = DATA_DIR / "train_labels.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"

# Create output directories
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
SUBMISSION_DIR.mkdir(exist_ok=True)

# Class labels mapping
LABEL_TO_IDX = {
    'apple': 0,
    'facebook': 1,
    'google': 2,
    'messenger': 3,
    'mozilla': 4,
    'samsung': 5,
    'whatsapp': 6
}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}
NUM_CLASSES = len(LABEL_TO_IDX)

# Image settings
IMG_SIZE = 72  # Original size
IMG_SIZE_MODEL = 224  # Size for pretrained models (resized)

# Training settings
BATCH_SIZE = 32
NUM_WORKERS = 8  # Set to 0 for Windows compatibility, increase on Linux
VAL_SPLIT = 0.2
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Device
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

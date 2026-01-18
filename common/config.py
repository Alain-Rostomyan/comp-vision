"""
Configuration settings for the project.
"""
import os
import torch
from pathlib import Path

class Config:
    # Project paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / '2-computer-vision-2025-b-sc-aidams-final-proj'
    
    TRAIN_DIR = DATA_DIR / 'train'
    TEST_DIR = DATA_DIR / 'test'
    TRAIN_LABELS = DATA_DIR / 'train_labels.csv'
    OUTPUT_DIR = BASE_DIR / 'outputs'
    
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Model parameters
    IMAGE_SIZE_BASELINE = 128
    IMAGE_SIZE_RESNET = 224
    IMAGE_SIZE_EFFICIENTNET = 224
    
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Class definitions
    CLASSES = ["apple", "facebook", "google", "messenger", "mozilla", "samsung", "whatsapp"]
    NUM_CLASSES = len(CLASSES)
    
    # Training settings
    SEED = 42
    WEIGHT_DECAY = 1e-4
    PATIENCE = 5
    VAL_SPLIT = 0.15

def get_device():
    """Returns the torch device."""
    return Config.DEVICE

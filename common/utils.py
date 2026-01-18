"""
Utility functions for training and reproducibility.
"""
import random
import numpy as np
import torch
import os
from common.config import Config

def set_seed(seed: int = Config.SEED):
    """
    Sets random seeds for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def save_checkpoint(model, optimizer, epoch, val_acc, filename):
    """
    Saves a model checkpoint.
    """
    path = Config.OUTPUT_DIR / filename
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }, path)
    print(f"Model saved to {path}")

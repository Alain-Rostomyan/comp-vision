"""
Training utilities for emoji classification.
"""
import time
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from src.config import DEVICE, SEED, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, MODEL_DIR, IDX_TO_LABEL


def set_seed(seed: int = SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    """Training class with validation and metrics tracking."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = DEVICE,
        model_name: str = "model"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model_name = model_name

        # History tracking
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }

        self.best_val_acc = 0.0
        self.best_epoch = 0

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self) -> Tuple[float, float, Dict[int, Dict]]:
        """Validate the model and compute per-class metrics."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        # Per-class tracking
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        all_preds = []
        all_labels = []

        for images, labels in tqdm(self.val_loader, desc="Validating", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Per-class stats
            for label, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        # Calculate per-class accuracy
        class_metrics = {}
        for cls_idx in sorted(class_total.keys()):
            cls_acc = 100. * class_correct[cls_idx] / class_total[cls_idx]
            class_metrics[cls_idx] = {
                'name': IDX_TO_LABEL[cls_idx],
                'correct': class_correct[cls_idx],
                'total': class_total[cls_idx],
                'accuracy': cls_acc
            }

        return epoch_loss, epoch_acc, class_metrics

    def train(
        self,
        num_epochs: int = NUM_EPOCHS,
        save_best: bool = True,
        early_stopping: int = 10
    ) -> Dict:
        """
        Full training loop.

        Args:
            num_epochs: Number of epochs to train
            save_best: Whether to save the best model
            early_stopping: Stop if no improvement for N epochs (0 to disable)
        """
        print(f"Training on {self.device}")
        print(f"Model: {self.model_name}")
        print("-" * 50)

        start_time = time.time()
        no_improve_count = 0

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc, class_metrics = self.validate()

            # Update scheduler
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)

            epoch_time = time.time() - epoch_start

            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")

            # Check for improvement
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                no_improve_count = 0

                if save_best:
                    self.save_model(f"{self.model_name}_best.pth")
                    print(f"  -> New best model saved! (Val Acc: {val_acc:.2f}%)")
            else:
                no_improve_count += 1

            # Early stopping
            if early_stopping > 0 and no_improve_count >= early_stopping:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

            print()

        total_time = time.time() - start_time
        print("-" * 50)
        print(f"Training completed in {total_time/60:.1f} minutes")
        print(f"Best Val Acc: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")

        # Final per-class metrics
        print("\nPer-class accuracy (validation set):")
        for cls_idx, metrics in class_metrics.items():
            print(f"  {metrics['name']:12s}: {metrics['accuracy']:5.2f}% "
                  f"({metrics['correct']}/{metrics['total']})")

        return {
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'total_time': total_time,
            'class_metrics': class_metrics
        }

    def save_model(self, filename: str):
        """Save model checkpoint."""
        path = MODEL_DIR / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'history': self.history
        }, path)

    def load_model(self, filename: str):
        """Load model checkpoint."""
        path = MODEL_DIR / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0)
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self.history = checkpoint.get('history', self.history)


def create_optimizer(
    model: nn.Module,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    optimizer_type: str = "adamw"
) -> torch.optim.Optimizer:
    """Create optimizer."""
    if optimizer_type.lower() == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    num_epochs: int = NUM_EPOCHS,
    scheduler_type: str = "cosine"
):
    """Create learning rate scheduler."""
    if scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type == "plateau":
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    elif scheduler_type == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")

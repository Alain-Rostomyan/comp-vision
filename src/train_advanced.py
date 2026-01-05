"""
Advanced training utilities with:
- Mixed precision training (AMP)
- Label smoothing
- Learning rate warmup
- Gradient accumulation
- Mixup/CutMix
- Cosine annealing with warmup
"""
import time
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.config import DEVICE, SEED, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, MODEL_DIR, IDX_TO_LABEL
from src.transforms_advanced import MixupCutmix


def set_seed(seed: int = SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(1)
        log_preds = F.log_softmax(pred, dim=1)

        # Create smoothed target
        with torch.no_grad():
            smooth_target = torch.zeros_like(log_preds)
            smooth_target.fill_(self.smoothing / (n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        loss = (-smooth_target * log_preds).sum(dim=1).mean()
        return loss


class MixupCrossEntropy(nn.Module):
    """Cross entropy loss for mixup/cutmix training."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(
        self,
        pred: torch.Tensor,
        target_a: torch.Tensor,
        target_b: torch.Tensor,
        lam: float
    ) -> torch.Tensor:
        if self.smoothing > 0:
            loss_fn = LabelSmoothingCrossEntropy(self.smoothing)
        else:
            loss_fn = nn.CrossEntropyLoss()

        return lam * loss_fn(pred, target_a) + (1 - lam) * loss_fn(pred, target_b)


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """Cosine annealing scheduler with warmup and restarts."""

    def __init__(
        self,
        optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 1e-6,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1
    ):
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma

        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch

        super().__init__(optimizer, last_epoch)
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                    for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) *
                    (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) /
                                  (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) *
                                           self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1),
                                     self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps *
                                                      (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** n
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class AdvancedTrainer:
    """Advanced training class with modern techniques."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        device: torch.device = DEVICE,
        model_name: str = "model",
        use_amp: bool = True,
        label_smoothing: float = 0.1,
        mixup_cutmix: Optional[MixupCutmix] = None,
        gradient_accumulation_steps: int = 1,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model_name = model_name
        self.use_amp = use_amp and device.type == "cuda"
        self.mixup_cutmix = mixup_cutmix
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Loss functions
        self.criterion = LabelSmoothingCrossEntropy(label_smoothing) if label_smoothing > 0 else nn.CrossEntropyLoss()
        self.mixup_criterion = MixupCrossEntropy(label_smoothing)

        # AMP scaler
        self.scaler = GradScaler() if self.use_amp else None

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
        """Train for one epoch with AMP and mixup support."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        self.optimizer.zero_grad()
        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Apply Mixup/CutMix
            use_mixup = self.mixup_cutmix is not None and self.model.training
            if use_mixup:
                images, labels_a, labels_b, lam = self.mixup_cutmix(images, labels)

            # Forward pass with AMP
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)

                if use_mixup:
                    loss = self.mixup_criterion(outputs, labels_a, labels_b, lam)
                else:
                    loss = self.criterion(outputs, labels)

                loss = loss / self.gradient_accumulation_steps

            # Backward pass with gradient scaling
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            # Statistics (for non-mixup accuracy)
            running_loss += loss.item() * self.gradient_accumulation_steps * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)

            if use_mixup:
                # Approximate accuracy for mixup
                correct += (lam * predicted.eq(labels_a).sum().item() +
                           (1 - lam) * predicted.eq(labels_b).sum().item())
            else:
                correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self) -> Tuple[float, float, Dict[int, Dict]]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        val_criterion = nn.CrossEntropyLoss()

        for images, labels in tqdm(self.val_loader, desc="Validating", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)

            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = val_criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for label, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

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
        early_stopping: int = 15
    ) -> Dict:
        """Full training loop."""
        print(f"Training on {self.device}")
        print(f"Model: {self.model_name}")
        print(f"AMP enabled: {self.use_amp}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print("-" * 50)

        start_time = time.time()
        no_improve_count = 0

        for epoch in range(num_epochs):
            epoch_start = time.time()

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, class_metrics = self.validate()

            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                self.scheduler.step()

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)

            epoch_time = time.time() - epoch_start

            print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                no_improve_count = 0

                if save_best:
                    self.save_model(f"{self.model_name}_best.pth")
                    print(f"  -> New best model saved! (Val Acc: {val_acc:.2f}%)")
            else:
                no_improve_count += 1

            if early_stopping > 0 and no_improve_count >= early_stopping:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

            print()

        total_time = time.time() - start_time
        print("-" * 50)
        print(f"Training completed in {total_time/60:.1f} minutes")
        print(f"Best Val Acc: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")

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
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0)
        self.best_epoch = checkpoint.get('best_epoch', 0)


def create_optimizer_with_param_groups(
    model: nn.Module,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    backbone_lr_mult: float = 0.1
) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer with different learning rates for backbone and head.
    """
    classifier_names = ['classifier', 'fc', 'head', 'linear']
    classifier_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if any(cn in name.lower() for cn in classifier_names):
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": lr * backbone_lr_mult},
        {"params": classifier_params, "lr": lr},
    ]

    return AdamW(param_groups, lr=lr, weight_decay=weight_decay)


def create_scheduler_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6
) -> CosineAnnealingWarmupRestarts:
    """Create cosine annealing scheduler with warmup."""
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    return CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=total_steps,
        max_lr=optimizer.param_groups[-1]['lr'],  # Use head LR as max
        min_lr=min_lr,
        warmup_steps=warmup_steps,
    )

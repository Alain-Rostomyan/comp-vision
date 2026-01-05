#!/usr/bin/env python3
"""
Ensemble training script for maximum performance.
Trains multiple models and combines their predictions.

Usage:
    python train_ensemble.py --quick    # Quick ensemble (2 models)
    python train_ensemble.py --full     # Full ensemble (4 models)
    python train_ensemble.py --custom convnext_small efficientnetv2_s swin_small_patch4_window7_224
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold

from src.config import (
    TRAIN_DIR, TEST_DIR, TRAIN_LABELS_PATH,
    DEVICE, SEED, VAL_SPLIT, NUM_WORKERS, MODEL_DIR, SUBMISSION_DIR
)
from src.dataset import EmojiDataset
from src.models_advanced import create_model, get_model_config, count_parameters
from src.transforms_advanced import get_train_transforms_wrapped, get_val_transforms_wrapped, MixupCutmix
from src.train_advanced import (
    AdvancedTrainer, set_seed,
    create_optimizer_with_param_groups, create_scheduler_with_warmup
)
from src.inference_advanced import (
    predict_with_tta, create_submission, load_model_for_inference,
    analyze_predictions, predict_ensemble_with_tta
)


# Ensemble configurations
QUICK_ENSEMBLE = [
    "convnext_small.fb_in1k",
    "tf_efficientnetv2_s.in21k_ft_in1k",
]

FULL_ENSEMBLE = [
    "convnext_small.fb_in1k",
    "tf_efficientnetv2_s.in21k_ft_in1k",
    "swin_small_patch4_window7_224.ms_in1k",
    "efficientnet_b3.ra2_in1k",
]

BEST_ENSEMBLE = [
    "convnext_small",
    "convnextv2_tiny",
    "efficientnetv2_s",
    "swin_small_patch4_window7_224",
    "eva02_small_patch14_224",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble Training for Emoji Classification")

    # Ensemble selection
    parser.add_argument('--quick', action='store_true', help='Use quick ensemble (2 models)')
    parser.add_argument('--full', action='store_true', help='Use full ensemble (4 models)')
    parser.add_argument('--best', action='store_true', help='Use best ensemble (5 models)')
    parser.add_argument('--custom', nargs='+', help='Custom list of models')

    # Training settings
    parser.add_argument('--epochs', type=int, default=40, help='Epochs per model')
    parser.add_argument('--batch-size', type=int, default=48, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

    # Augmentation
    parser.add_argument('--heavy-aug', action='store_true', help='Use heavy augmentation')
    parser.add_argument('--no-mixup', action='store_true', help='Disable Mixup/CutMix')

    # Inference
    parser.add_argument('--tta', type=int, default=5, help='TTA augmentations')
    parser.add_argument('--skip-training', action='store_true', help='Skip training, only run inference')

    # K-Fold
    parser.add_argument('--kfold', type=int, default=0, help='K-fold cross validation (0=disabled)')

    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')

    return parser.parse_args()


def train_single_model(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    args,
    fold: int = 0
) -> dict:
    """Train a single model."""

    model_config = get_model_config(model_name)
    img_size = model_config['img_size']
    save_name = f"{model_name}_fold{fold}" if args.kfold > 0 else model_name
    checkpoint_path = MODEL_DIR / f"{save_name}_best.pth"

    # Check if model already exists
    if checkpoint_path.exists():
        print(f"\n{'='*60}")
        print(f"SKIPPING: {model_name} (checkpoint exists)")
        print(f"{'='*60}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        return {
            'model_name': model_name,
            'fold': fold,
            'best_val_acc': checkpoint.get('best_val_acc', 95.0),
            'best_epoch': checkpoint.get('best_epoch', 0),
            'img_size': img_size,
            'checkpoint_path': checkpoint_path
        }

    print(f"\n{'='*60}")
    print(f"Training: {model_name} (fold {fold})")
    print(f"Image size: {img_size}")
    print(f"{'='*60}")

    # Create transforms
    train_transform = get_train_transforms_wrapped(img_size, heavy=args.heavy_aug)
    val_transform = get_val_transforms_wrapped(img_size)

    # Create datasets
    train_dataset = EmojiDataset(image_dir=TRAIN_DIR, labels_df=train_df, transform=train_transform)
    val_dataset = EmojiDataset(image_dir=TRAIN_DIR, labels_df=val_df, transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # Create model
    model = create_model(
        model_name, pretrained=True,
        drop_rate=model_config['drop_rate'],
        drop_path_rate=model_config['drop_path_rate']
    )
    print(f"Parameters: {count_parameters(model):,}")

    # Create optimizer and scheduler
    optimizer = create_optimizer_with_param_groups(model, lr=args.lr, backbone_lr_mult=0.1)
    scheduler = create_scheduler_with_warmup(
        optimizer, args.epochs, len(train_loader), warmup_epochs=3
    )

    # Mixup/CutMix
    mixup_cutmix = None if args.no_mixup else MixupCutmix(prob=0.5)

    # Train
    trainer = AdvancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        model_name=save_name,
        use_amp=True,
        label_smoothing=0.1,
        mixup_cutmix=mixup_cutmix,
        gradient_accumulation_steps=1
    )

    results = trainer.train(num_epochs=args.epochs, save_best=True, early_stopping=12)

    return {
        'model_name': model_name,
        'fold': fold,
        'best_val_acc': results['best_val_acc'],
        'best_epoch': results['best_epoch'],
        'img_size': img_size,
        'checkpoint_path': MODEL_DIR / f"{save_name}_best.pth"
    }


def run_ensemble_inference(
    model_results: list,
    args
) -> tuple:
    """Run ensemble inference with TTA."""

    print("\n" + "=" * 60)
    print("ENSEMBLE INFERENCE WITH TTA")
    print("=" * 60)

    # Get test images
    test_images = sorted(TEST_DIR.glob("*.png"))
    test_ids = [img.stem for img in test_images]
    print(f"Test samples: {len(test_images)}")

    # Load models
    models = []
    img_sizes = []
    weights = []

    for result in model_results:
        model = create_model(result['model_name'], pretrained=False)
        model = load_model_for_inference(model, result['checkpoint_path'], DEVICE)
        models.append(model)
        img_sizes.append(result['img_size'])
        # Weight by validation accuracy
        weights.append(result['best_val_acc'])

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    print(f"\nModel weights: {weights}")

    # Ensemble prediction with TTA
    predictions, probabilities = predict_ensemble_with_tta(
        models=models,
        image_paths=test_images,
        img_sizes=img_sizes,
        num_tta=args.tta,
        batch_size=args.batch_size,
        device=DEVICE,
        model_weights=weights,
        use_amp=True
    )

    return predictions, probabilities, test_ids


def main():
    args = parse_args()
    set_seed(args.seed)

    # Select models
    if args.custom:
        models = args.custom
    elif args.best:
        models = BEST_ENSEMBLE
    elif args.full:
        models = FULL_ENSEMBLE
    else:
        models = QUICK_ENSEMBLE

    print("=" * 60)
    print("ENSEMBLE TRAINING")
    print("=" * 60)
    print(f"Models: {models}")
    print(f"Epochs per model: {args.epochs}")
    print(f"K-Fold: {args.kfold if args.kfold > 0 else 'disabled'}")
    print(f"TTA augmentations: {args.tta}")

    # Load data
    train_labels = pd.read_csv(TRAIN_LABELS_PATH)
    print(f"\nTotal samples: {len(train_labels)}")

    model_results = []

    if not args.skip_training:
        if args.kfold > 0:
            # K-Fold training
            skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)

            for model_name in models:
                for fold, (train_idx, val_idx) in enumerate(skf.split(train_labels, train_labels['Label'])):
                    train_df = train_labels.iloc[train_idx]
                    val_df = train_labels.iloc[val_idx]
                    result = train_single_model(model_name, train_df, val_df, args, fold)
                    model_results.append(result)
        else:
            # Single split training
            train_df, val_df = train_test_split(
                train_labels, test_size=VAL_SPLIT,
                stratify=train_labels['Label'], random_state=args.seed
            )

            for model_name in models:
                result = train_single_model(model_name, train_df, val_df, args)
                model_results.append(result)

        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        for result in model_results:
            print(f"{result['model_name']} (fold {result['fold']}): "
                  f"{result['best_val_acc']:.2f}% (epoch {result['best_epoch']})")

    else:
        # Load existing checkpoints
        for model_name in models:
            checkpoint_path = MODEL_DIR / f"{model_name}_best.pth"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
                model_results.append({
                    'model_name': model_name,
                    'fold': 0,
                    'best_val_acc': checkpoint.get('best_val_acc', 95.0),
                    'best_epoch': checkpoint.get('best_epoch', 0),
                    'img_size': get_model_config(model_name)['img_size'],
                    'checkpoint_path': checkpoint_path
                })
                print(f"Loaded: {model_name} ({checkpoint.get('best_val_acc', 0):.2f}%)")
            else:
                print(f"Warning: Checkpoint not found for {model_name}")

    if not model_results:
        print("No trained models available!")
        return

    # Run ensemble inference
    predictions, probabilities, test_ids = run_ensemble_inference(model_results, args)

    # Analyze predictions
    analysis = analyze_predictions(probabilities, predictions)
    print(f"\nPrediction Analysis:")
    print(f"  Mean confidence: {analysis['mean_confidence']:.4f}")
    print(f"  Low confidence (<0.7): {analysis['low_confidence_count']}")
    print(f"  Class distribution:")
    for cls, count in analysis['class_distribution'].items():
        print(f"    {cls}: {count}")

    # Create submission
    model_suffix = "_".join([m.split("_")[0] for m in models[:3]])
    submission_name = f"submission_ensemble_{model_suffix}_tta{args.tta}.csv"
    create_submission(predictions, test_ids, submission_name, use_labels=True)

    # Calculate ensemble validation accuracy estimate
    avg_val_acc = np.mean([r['best_val_acc'] for r in model_results])
    print(f"\nAverage validation accuracy: {avg_val_acc:.2f}%")
    print(f"(Ensemble typically adds +0.5-1.0% over average)")

    print("\n" + "=" * 60)
    print("ENSEMBLE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

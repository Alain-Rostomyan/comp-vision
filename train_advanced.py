#!/usr/bin/env python3
"""
Advanced training script for emoji classification.
Includes all optimizations for maximum performance.

Usage:
    python train_advanced.py --model convnext_small --epochs 50 --batch-size 64
    python train_advanced.py --model efficientnetv2_s --epochs 40 --tta 5
    python train_advanced.py --model swin_small_patch4_window7_224 --heavy-aug
"""
import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.config import (
    TRAIN_DIR, TEST_DIR, TRAIN_LABELS_PATH,
    DEVICE, SEED, VAL_SPLIT, NUM_WORKERS, MODEL_DIR
)
from src.dataset import EmojiDataset
from src.models_advanced import (
    create_model, get_model_config, count_parameters,
    RECOMMENDED_MODELS, list_available_models
)
from src.transforms_advanced import (
    get_train_transforms_wrapped, get_val_transforms_wrapped, MixupCutmix
)
from src.train_advanced import (
    AdvancedTrainer, set_seed,
    create_optimizer_with_param_groups, create_scheduler_with_warmup
)
from src.inference_advanced import (
    predict_with_tta, create_submission, load_model_for_inference,
    analyze_predictions
)


def parse_args():
    parser = argparse.ArgumentParser(description="Advanced Emoji Classification Training")

    # Model settings
    parser.add_argument('--model', type=str, default='convnext_small',
                        help=f'Model architecture. Options: {", ".join(list_available_models())}')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Do not use pretrained weights')

    # Training settings
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--backbone-lr-mult', type=float, default=0.1,
                        help='Learning rate multiplier for backbone')

    # Augmentation settings
    parser.add_argument('--heavy-aug', action='store_true',
                        help='Use heavy augmentation')
    parser.add_argument('--no-mixup', action='store_true',
                        help='Disable Mixup/CutMix')
    parser.add_argument('--mixup-prob', type=float, default=0.5,
                        help='Probability of applying Mixup/CutMix')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing factor')

    # Training optimizations
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable automatic mixed precision')
    parser.add_argument('--grad-accum', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--early-stopping', type=int, default=15,
                        help='Early stopping patience (0 to disable)')

    # Inference settings
    parser.add_argument('--tta', type=int, default=5,
                        help='Number of TTA augmentations (1=no TTA)')
    parser.add_argument('--generate-submission', action='store_true',
                        help='Generate submission after training')

    # Other settings
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Random seed')
    parser.add_argument('--num-workers', type=int, default=NUM_WORKERS,
                        help='Number of data loading workers')
    parser.add_argument('--img-size', type=int, default=None,
                        help='Image size (auto-detected if not specified)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    print("=" * 60)
    print("ADVANCED EMOJI CLASSIFICATION TRAINING")
    print("=" * 60)

    # Get model configuration
    model_config = get_model_config(args.model)
    img_size = args.img_size or model_config['img_size']

    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Image size: {img_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Heavy augmentation: {args.heavy_aug}")
    print(f"  Mixup/CutMix: {not args.no_mixup}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  AMP: {not args.no_amp}")
    print(f"  TTA augmentations: {args.tta}")
    print(f"  Device: {DEVICE}")

    # Load data
    print("\nLoading data...")
    train_labels = pd.read_csv(TRAIN_LABELS_PATH)
    print(f"Total training samples: {len(train_labels)}")

    # Stratified split
    train_df, val_df = train_test_split(
        train_labels,
        test_size=VAL_SPLIT,
        stratify=train_labels['Label'],
        random_state=args.seed
    )
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    # Create transforms
    train_transform = get_train_transforms_wrapped(img_size, heavy=args.heavy_aug)
    val_transform = get_val_transforms_wrapped(img_size)

    # Create datasets
    train_dataset = EmojiDataset(
        image_dir=TRAIN_DIR,
        labels_df=train_df,
        transform=train_transform
    )
    val_dataset = EmojiDataset(
        image_dir=TRAIN_DIR,
        labels_df=val_df,
        transform=val_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Create model
    print(f"\nCreating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=not args.no_pretrained,
        drop_rate=model_config['drop_rate'],
        drop_path_rate=model_config['drop_path_rate']
    )
    print(f"Parameters: {count_parameters(model):,}")

    # Create optimizer with differential learning rates
    optimizer = create_optimizer_with_param_groups(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        backbone_lr_mult=args.backbone_lr_mult
    )

    # Create scheduler with warmup
    scheduler = create_scheduler_with_warmup(
        optimizer,
        num_epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        warmup_epochs=args.warmup_epochs
    )

    # Create Mixup/CutMix
    mixup_cutmix = None if args.no_mixup else MixupCutmix(prob=args.mixup_prob)

    # Create trainer
    trainer = AdvancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        model_name=args.model,
        use_amp=not args.no_amp,
        label_smoothing=args.label_smoothing,
        mixup_cutmix=mixup_cutmix,
        gradient_accumulation_steps=args.grad_accum
    )

    # Train
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    results = trainer.train(
        num_epochs=args.epochs,
        save_best=True,
        early_stopping=args.early_stopping
    )

    # Generate submission if requested
    if args.generate_submission:
        print("\n" + "=" * 60)
        print("GENERATING SUBMISSION")
        print("=" * 60)

        # Load best model
        checkpoint_path = MODEL_DIR / f"{args.model}_best.pth"
        model = create_model(args.model, pretrained=False)
        model = load_model_for_inference(model, checkpoint_path, DEVICE)

        # Get test images
        test_images = sorted(TEST_DIR.glob("*.png"))
        test_ids = [img.stem for img in test_images]
        print(f"Test samples: {len(test_images)}")

        # Predict with TTA
        predictions, probabilities = predict_with_tta(
            model=model,
            image_paths=test_images,
            img_size=img_size,
            num_tta=args.tta,
            batch_size=args.batch_size,
            device=DEVICE,
            use_amp=not args.no_amp
        )

        # Analyze predictions
        analysis = analyze_predictions(probabilities, predictions)
        print(f"\nPrediction Analysis:")
        print(f"  Mean confidence: {analysis['mean_confidence']:.4f}")
        print(f"  Low confidence (<0.7): {analysis['low_confidence_count']}")
        print(f"  High confidence (>0.9): {analysis['high_confidence_count']}")
        print(f"  Class distribution:")
        for cls, count in analysis['class_distribution'].items():
            print(f"    {cls}: {count}")

        # Create submission
        tta_suffix = f"_tta{args.tta}" if args.tta > 1 else ""
        submission_name = f"submission_{args.model}{tta_suffix}.csv"
        create_submission(predictions, test_ids, submission_name, use_labels=True)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation accuracy: {results['best_val_acc']:.2f}%")
    print(f"Best epoch: {results['best_epoch']}")
    print(f"Total training time: {results['total_time']/60:.1f} minutes")

    return results


if __name__ == "__main__":
    main()

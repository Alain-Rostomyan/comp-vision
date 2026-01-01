"""
Main training script for emoji classification.

Usage:
    python train_model.py --model efficientnet_b0 --epochs 30
    python train_model.py --model custom_cnn --epochs 50 --img-size 72
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn

from src.config import (
    DEVICE, SEED, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE,
    IMG_SIZE, IMG_SIZE_MODEL, NUM_CLASSES
)
from src.dataset import (
    load_train_data, get_train_val_split, create_dataloaders,
    create_test_dataloader, get_class_weights
)
from src.transforms import (
    get_train_transforms, get_val_transforms,
    get_train_transforms_simple, get_val_transforms_simple
)
from src.models import get_model, count_parameters
from src.train import (
    set_seed, Trainer, create_optimizer, create_scheduler
)
from src.inference import predict, create_submission


def parse_args():
    parser = argparse.ArgumentParser(description="Train emoji classifier")

    parser.add_argument(
        "--model", type=str, default="efficientnet_b0",
        choices=["custom_cnn", "light_cnn", "efficientnet_b0", "efficientnet_b1",
                 "resnet18", "resnet34", "mobilenet"],
        help="Model architecture to use"
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=None, help="Image size (auto-detected)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--no-pretrained", action="store_true", help="Don't use pretrained weights")
    parser.add_argument("--early-stopping", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--use-class-weights", action="store_true", help="Use class weights")
    parser.add_argument("--generate-submission", action="store_true", help="Generate submission after training")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")
    print(f"Device: {DEVICE}")
    print("=" * 50)

    # Determine image size based on model
    if args.img_size is not None:
        img_size = args.img_size
    elif args.model in ["custom_cnn", "light_cnn"]:
        img_size = IMG_SIZE  # 72 for custom models
    else:
        img_size = IMG_SIZE_MODEL  # 224 for pretrained models

    print(f"Using image size: {img_size}")

    # Load data
    print("\nLoading data...")
    df = load_train_data()
    train_df, val_df = get_train_val_split(df, seed=args.seed)
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    # Create transforms
    if args.model in ["custom_cnn", "light_cnn"]:
        train_transform = get_train_transforms_simple(img_size)
        val_transform = get_val_transforms_simple(img_size)
    else:
        train_transform = get_train_transforms(img_size)
        val_transform = get_val_transforms(img_size)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_df, val_df,
        train_transform, val_transform,
        batch_size=args.batch_size
    )

    # Create model
    print(f"\nCreating model: {args.model}")
    pretrained = not args.no_pretrained and args.model not in ["custom_cnn", "light_cnn"]
    model = get_model(args.model, NUM_CLASSES, pretrained=pretrained)
    print(f"Parameters: {count_parameters(model):,}")

    # Loss function
    if args.use_class_weights:
        class_weights = get_class_weights(train_df).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using class weights for imbalanced data")
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer and scheduler
    optimizer = create_optimizer(model, lr=args.lr)
    scheduler = create_scheduler(optimizer, num_epochs=args.epochs)

    # Train
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50 + "\n")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        model_name=args.model
    )

    results = trainer.train(
        num_epochs=args.epochs,
        save_best=True,
        early_stopping=args.early_stopping
    )

    # Generate submission if requested
    if args.generate_submission:
        print("\n" + "=" * 50)
        print("Generating submission...")
        print("=" * 50)

        # Load best model
        from src.config import MODEL_DIR
        checkpoint = torch.load(MODEL_DIR / f"{args.model}_best.pth", map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Create test dataloader
        test_loader, test_ids = create_test_dataloader(val_transform, batch_size=args.batch_size)

        # Make predictions
        predictions, _ = predict(model, test_loader, DEVICE)

        # Create submission
        submission = create_submission(
            predictions, test_ids,
            filename=f"submission_{args.model}.csv",
            use_labels=True
        )

        print(f"\nSubmission created with {len(submission)} predictions")

    print("\nDone!")


if __name__ == "__main__":
    main()

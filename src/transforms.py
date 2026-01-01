"""
Image transforms and augmentations for training.
"""
import torchvision.transforms as T
from src.config import IMG_SIZE, IMG_SIZE_MODEL


def get_train_transforms(img_size: int = IMG_SIZE_MODEL):
    """
    Training transforms with data augmentation.

    Augmentations:
    - Resize to model input size
    - Random horizontal flip
    - Random rotation
    - Color jitter (brightness, contrast, saturation)
    - Normalize with ImageNet stats (for pretrained models)
    """
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms(img_size: int = IMG_SIZE_MODEL):
    """
    Validation/Test transforms (no augmentation).
    """
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_train_transforms_simple(img_size: int = IMG_SIZE):
    """
    Simple training transforms for custom CNN (no ImageNet normalization).
    """
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def get_val_transforms_simple(img_size: int = IMG_SIZE):
    """
    Simple validation transforms for custom CNN.
    """
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

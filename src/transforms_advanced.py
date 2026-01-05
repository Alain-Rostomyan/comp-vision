"""
Advanced image transforms and augmentations using Albumentations.
Includes RandAugment, Mixup, CutMix for improved model generalization.
"""
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T
from PIL import Image
from typing import Tuple, Optional

from src.config import IMG_SIZE_MODEL


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_advanced_train_transforms(img_size: int = IMG_SIZE_MODEL):
    """
    Advanced training transforms with strong augmentation.
    Uses Albumentations for better augmentation variety.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.85, 1.15),
            rotate=(-15, 15),
            p=0.5
        ),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
        ], p=0.7),
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.05), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0),
            A.ElasticTransform(alpha=1, sigma=30, p=1.0),
        ], p=0.2),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(img_size // 16, img_size // 8),
            hole_width_range=(img_size // 16, img_size // 8),
            p=0.3
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def get_heavy_train_transforms(img_size: int = IMG_SIZE_MODEL):
    """
    Heavy augmentation for maximum regularization.
    Use with larger models to prevent overfitting.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Affine(
            translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
            scale=(0.8, 1.2),
            rotate=(-20, 20),
            p=0.6
        ),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=30, p=0.5),
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.1), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.4),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.15, p=1.0),
            A.ElasticTransform(alpha=1.5, sigma=40, p=1.0),
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=1.0),
        ], p=0.3),
        A.CoarseDropout(
            num_holes_range=(2, 12),
            hole_height_range=(img_size // 12, img_size // 6),
            hole_width_range=(img_size // 12, img_size // 6),
            p=0.4
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def get_val_transforms_album(img_size: int = IMG_SIZE_MODEL):
    """Validation transforms using Albumentations (no augmentation)."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def get_tta_transforms(img_size: int = IMG_SIZE_MODEL):
    """
    Test-Time Augmentation transforms.
    Returns a list of transforms to apply during inference.
    """
    base_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

    hflip_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

    rotate_transforms = []
    for angle in [-10, 10]:
        rotate_transforms.append(A.Compose([
            A.Resize(img_size, img_size),
            A.Rotate(limit=(angle, angle), border_mode=0, p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ]))

    scale_transforms = []
    for scale in [0.9, 1.1]:
        scale_transforms.append(A.Compose([
            A.Resize(int(img_size * scale), int(img_size * scale)),
            A.CenterCrop(img_size, img_size) if scale > 1 else A.PadIfNeeded(img_size, img_size, border_mode=0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ]))

    return [base_transform, hflip_transform] + rotate_transforms


class MixupCutmix:
    """
    Mixup and CutMix data augmentation for training.
    """
    def __init__(self, mixup_alpha: float = 0.4, cutmix_alpha: float = 1.0, prob: float = 0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob

    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply Mixup or CutMix to a batch.

        Returns:
            mixed_images: Mixed batch of images
            labels_a: Original labels
            labels_b: Labels to mix with
            lam: Mixing coefficient
        """
        if np.random.random() > self.prob:
            return images, labels, labels, 1.0

        batch_size = images.size(0)
        indices = torch.randperm(batch_size)

        # Choose between Mixup and CutMix
        if np.random.random() > 0.5:
            # Mixup
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            mixed_images = lam * images + (1 - lam) * images[indices]
        else:
            # CutMix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
            mixed_images = images.clone()
            mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images[indices, :, bbx1:bbx2, bby1:bby2]
            # Adjust lambda based on actual area
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))

        return mixed_images, labels, labels[indices], lam

    def _rand_bbox(self, size, lam):
        """Generate random bounding box for CutMix."""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


class AlbumentationsWrapper:
    """Wrapper to use Albumentations transforms with PIL images."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
        augmented = self.transform(image=image)
        return augmented['image']


def get_train_transforms_wrapped(img_size: int = IMG_SIZE_MODEL, heavy: bool = False):
    """Get training transforms wrapped for use with standard PyTorch datasets."""
    if heavy:
        transform = get_heavy_train_transforms(img_size)
    else:
        transform = get_advanced_train_transforms(img_size)
    return AlbumentationsWrapper(transform)


def get_val_transforms_wrapped(img_size: int = IMG_SIZE_MODEL):
    """Get validation transforms wrapped for use with standard PyTorch datasets."""
    return AlbumentationsWrapper(get_val_transforms_album(img_size))

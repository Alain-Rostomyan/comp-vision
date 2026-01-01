"""
Model architectures for emoji classification.
Includes custom CNN and transfer learning models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional

from src.config import NUM_CLASSES, IMG_SIZE, IMG_SIZE_MODEL


class CustomCNN(nn.Module):
    """
    Custom CNN architecture for emoji classification.
    Designed for 72x72 input images.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()

        # Convolutional blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 72 -> 36
            nn.Dropout2d(0.25)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 36 -> 18
            nn.Dropout2d(0.25)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 18 -> 9
            nn.Dropout2d(0.25)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)
        return x


class LightCNN(nn.Module):
    """
    Lighter CNN for faster training/inference.
    Good for quick experiments.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 72 -> 36

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 36 -> 18

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 18 -> 9

            # Global pooling
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_efficientnet(
    model_name: str = "efficientnet_b0",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    dropout: float = 0.3
) -> nn.Module:
    """
    Create EfficientNet model using timm library.

    Args:
        model_name: EfficientNet variant (b0, b1, b2, etc.)
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate for classifier
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout
    )
    return model


def create_resnet(
    model_name: str = "resnet18",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True
) -> nn.Module:
    """
    Create ResNet model using timm library.

    Args:
        model_name: ResNet variant (resnet18, resnet34, resnet50, etc.)
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model


def create_mobilenet(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True
) -> nn.Module:
    """
    Create MobileNetV3 model - lightweight and fast.
    Good for CPU training.
    """
    model = timm.create_model(
        "mobilenetv3_small_100",
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model


def get_model(
    model_type: str,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: One of 'custom_cnn', 'light_cnn', 'efficientnet_b0',
                   'efficientnet_b1', 'resnet18', 'resnet34', 'mobilenet'
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (for transfer learning)
    """
    model_type = model_type.lower()

    if model_type == 'custom_cnn':
        return CustomCNN(num_classes=num_classes, **kwargs)
    elif model_type == 'light_cnn':
        return LightCNN(num_classes=num_classes, **kwargs)
    elif model_type.startswith('efficientnet'):
        return create_efficientnet(model_type, num_classes, pretrained, **kwargs)
    elif model_type.startswith('resnet'):
        return create_resnet(model_type, num_classes, pretrained)
    elif model_type == 'mobilenet':
        return create_mobilenet(num_classes, pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

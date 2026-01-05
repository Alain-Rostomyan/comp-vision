"""
Advanced model architectures for emoji classification.
Includes ConvNeXt, EfficientNetV2, Swin Transformer, and more.
"""
import torch
import torch.nn as nn
import timm
from typing import Optional, List, Dict

from src.config import NUM_CLASSES


def create_model(
    model_name: str,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
) -> nn.Module:
    """
    Create a model using timm library.

    Supported models:
    - efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4
    - efficientnetv2_s, efficientnetv2_m
    - convnext_tiny, convnext_small, convnext_base
    - swin_tiny_patch4_window7_224, swin_small_patch4_window7_224
    - vit_small_patch16_224, vit_base_patch16_224
    - resnet18, resnet34, resnet50
    - eva02_tiny_patch14_224, eva02_small_patch14_224
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )
    return model


# Model configurations with recommended settings
MODEL_CONFIGS = {
    # EfficientNet family
    "efficientnet_b0": {"img_size": 224, "drop_rate": 0.2, "drop_path_rate": 0.0},
    "efficientnet_b1": {"img_size": 240, "drop_rate": 0.2, "drop_path_rate": 0.0},
    "efficientnet_b2": {"img_size": 260, "drop_rate": 0.3, "drop_path_rate": 0.0},
    "efficientnet_b3": {"img_size": 300, "drop_rate": 0.3, "drop_path_rate": 0.0},
    "efficientnet_b4": {"img_size": 380, "drop_rate": 0.4, "drop_path_rate": 0.0},

    # EfficientNetV2 family (better than V1)
    "efficientnetv2_s": {"img_size": 300, "drop_rate": 0.2, "drop_path_rate": 0.2},
    "efficientnetv2_m": {"img_size": 384, "drop_rate": 0.3, "drop_path_rate": 0.2},

    # ConvNeXt family (very strong, modern architecture)
    "convnext_tiny": {"img_size": 224, "drop_rate": 0.0, "drop_path_rate": 0.1},
    "convnext_small": {"img_size": 224, "drop_rate": 0.0, "drop_path_rate": 0.4},
    "convnext_base": {"img_size": 224, "drop_rate": 0.0, "drop_path_rate": 0.5},
    "convnextv2_tiny": {"img_size": 224, "drop_rate": 0.0, "drop_path_rate": 0.1},
    "convnextv2_base": {"img_size": 224, "drop_rate": 0.0, "drop_path_rate": 0.5},

    # Swin Transformer family
    "swin_tiny_patch4_window7_224": {"img_size": 224, "drop_rate": 0.0, "drop_path_rate": 0.2},
    "swin_small_patch4_window7_224": {"img_size": 224, "drop_rate": 0.0, "drop_path_rate": 0.3},
    "swin_base_patch4_window7_224": {"img_size": 224, "drop_rate": 0.0, "drop_path_rate": 0.5},

    # Vision Transformer family
    "vit_small_patch16_224": {"img_size": 224, "drop_rate": 0.0, "drop_path_rate": 0.1},
    "vit_base_patch16_224": {"img_size": 224, "drop_rate": 0.0, "drop_path_rate": 0.1},
    "deit_small_patch16_224": {"img_size": 224, "drop_rate": 0.0, "drop_path_rate": 0.1},
    "deit3_small_patch16_224": {"img_size": 224, "drop_rate": 0.0, "drop_path_rate": 0.1},

    # EVA-02 (very strong, state-of-the-art)
    "eva02_tiny_patch14_224": {"img_size": 224, "drop_rate": 0.0, "drop_path_rate": 0.0},
    "eva02_small_patch14_224": {"img_size": 224, "drop_rate": 0.0, "drop_path_rate": 0.0},
    "eva02_base_patch14_224": {"img_size": 224, "drop_rate": 0.0, "drop_path_rate": 0.0},

    # ResNet family (baseline)
    "resnet18": {"img_size": 224, "drop_rate": 0.0, "drop_path_rate": 0.0},
    "resnet34": {"img_size": 224, "drop_rate": 0.0, "drop_path_rate": 0.0},
    "resnet50": {"img_size": 224, "drop_rate": 0.0, "drop_path_rate": 0.0},

    # CAFormer / MetaFormer
    "caformer_s18": {"img_size": 224, "drop_rate": 0.0, "drop_path_rate": 0.2},
    "caformer_b36": {"img_size": 224, "drop_rate": 0.0, "drop_path_rate": 0.3},

    # MaxViT
    "maxvit_tiny_tf_224": {"img_size": 224, "drop_rate": 0.0, "drop_path_rate": 0.2},
}


def get_model_config(model_name: str) -> Dict:
    """Get recommended configuration for a model."""
    # Strip any suffix like .fb_in1k, .in21k_ft_in1k, etc.
    base_name = model_name.split('.')[0]

    # Try exact match first
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]

    if base_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[base_name]

    # Try partial match
    for key in MODEL_CONFIGS:
        if key in base_name or base_name in key:
            return MODEL_CONFIGS[key]

    # Default config
    return {"img_size": 224, "drop_rate": 0.2, "drop_path_rate": 0.1}


def get_model_with_config(
    model_name: str,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> tuple:
    """
    Create a model with its recommended configuration.

    Returns:
        model: The created model
        config: Model configuration dict with img_size, drop_rate, etc.
    """
    config = get_model_config(model_name)
    model = create_model(
        model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        drop_rate=config["drop_rate"],
        drop_path_rate=config["drop_path_rate"],
    )
    return model, config


class ModelEnsemble(nn.Module):
    """Ensemble of multiple models for improved predictions."""

    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for model, weight in zip(self.models, self.weights):
            out = model(x)
            outputs.append(weight * torch.softmax(out, dim=1))
        return torch.stack(outputs).sum(dim=0)


def freeze_backbone(model: nn.Module, freeze_ratio: float = 0.7):
    """
    Freeze a portion of the model's backbone for transfer learning.

    Args:
        model: The model to freeze
        freeze_ratio: Fraction of layers to freeze (0.0-1.0)
    """
    params = list(model.parameters())
    num_freeze = int(len(params) * freeze_ratio)
    for param in params[:num_freeze]:
        param.requires_grad = False


def unfreeze_all(model: nn.Module):
    """Unfreeze all parameters in the model."""
    for param in model.parameters():
        param.requires_grad = True


def get_param_groups(model: nn.Module, base_lr: float, backbone_lr_mult: float = 0.1):
    """
    Get parameter groups with different learning rates.
    Backbone layers get lower LR, classifier gets full LR.

    Args:
        model: The model
        base_lr: Base learning rate for classifier
        backbone_lr_mult: Multiplier for backbone learning rate
    """
    # Try to identify classifier parameters
    classifier_names = ['classifier', 'fc', 'head', 'linear']
    classifier_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if any(cn in name.lower() for cn in classifier_names):
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    return [
        {"params": backbone_params, "lr": base_lr * backbone_lr_mult},
        {"params": classifier_params, "lr": base_lr},
    ]


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in the model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def list_available_models() -> List[str]:
    """List all available model configurations."""
    return list(MODEL_CONFIGS.keys())


# Recommended models for emoji classification (ordered by expected performance)
RECOMMENDED_MODELS = [
    "convnext_small",       # Strong, efficient
    "convnext_tiny",        # Good balance
    "efficientnetv2_s",     # Fast, strong
    "swin_small_patch4_window7_224",  # Transformer, strong
    "eva02_small_patch14_224",  # State-of-the-art
    "efficientnet_b3",      # Larger EfficientNet
    "deit3_small_patch16_224",  # Efficient ViT
]

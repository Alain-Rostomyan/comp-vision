"""
Model definitions for Emoji Classification.
"""
import torch.nn as nn
from torchvision import models

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=7, pretrained=False):
        super(ResNet18Classifier, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class EfficientNetB1Classifier(nn.Module):
    def __init__(self, num_classes=7, pretrained=False):
        super(EfficientNetB1Classifier, self).__init__()
        # Use simple creation for now, as timm might not be installed or needed
        # But if the project uses torchvision models, let's stick to that if possible.
        # However, EfficientNet B1 is available in newer torchvision.
        # If the original code used models.efficientnet_b1, we use that.
        self.model = models.efficientnet_b1(weights='IMAGENET1K_V1' if pretrained else None)
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

def get_model(name: str, num_classes: int = 7, pretrained: bool = False):
    """Factory method to get model by name."""
    if name == "resnet18":
        return ResNet18Classifier(num_classes, pretrained)
    elif name == "efficientnet_b1":
        return EfficientNetB1Classifier(num_classes, pretrained)
    else:
        raise ValueError(f"Unknown model name: {name}")

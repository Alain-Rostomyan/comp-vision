"""
Day 5 - Test-Time Augmentation (TTA)
Apply augmentations at test time and average predictions for better accuracy
Target: 96-97.5% by leveraging TTA
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configuration
class Config:
    TEST_DIR = 'test'
    IMAGE_SIZE = 256
    BATCH_SIZE = 16
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CLASSES = ["apple", "facebook", "google", "messenger", "mozilla", "samsung", "whatsapp"]
    NUM_CLASSES = len(CLASSES)
    
    # TTA settings
    TTA_TRANSFORMS = 8  # Number of augmented versions per image

# Model definitions (same as Day 4)
class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNet18Classifier, self).__init__()
        self.model = models.resnet18(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class EfficientNetB1Classifier(nn.Module):
    def __init__(self, num_classes=7):
        super(EfficientNetB1Classifier, self).__init__()
        self.model = models.efficientnet_b1(pretrained=False)
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

# TTA Transforms
def get_tta_transforms(image_size=256):
    """
    Create a list of TTA transforms
    Each transform is a different augmentation that could help the model
    """
    base_transform = [
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    
    tta_transforms = [
        # Original image
        A.Compose(base_transform),
        
        # Horizontal flip
        A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        
        # Vertical flip
        A.Compose([
            A.Resize(image_size, image_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        
        # Both flips
        A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        
        # Slight rotation
        A.Compose([
            A.Resize(image_size, image_size),
            A.Rotate(limit=15, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        
        # Brightness adjustment
        A.Compose([
            A.Resize(image_size, image_size),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        
        # Scale
        A.Compose([
            A.Resize(image_size, image_size),
            A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.1, rotate_limit=0, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        
        # Color jitter
        A.Compose([
            A.Resize(image_size, image_size),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    ]
    
    return tta_transforms

class TTADataset(Dataset):
    """Dataset that returns multiple augmented versions of each image"""
    def __init__(self, image_dir, tta_transforms):
        self.image_dir = image_dir
        self.tta_transforms = tta_transforms
        self.image_ids = sorted([f.split('.')[0] for f in os.listdir(image_dir) if f.endswith('.png')])
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{img_id}.png")
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Apply all TTA transforms
        augmented_images = []
        for transform in self.tta_transforms:
            aug = transform(image=image)
            augmented_images.append(aug['image'])
        
        # Stack all augmented versions
        return torch.stack(augmented_images), img_id

def predict_with_tta(model, dataloader, device):
    """
    Make predictions using Test-Time Augmentation
    Returns averaged probabilities across all augmentations
    """
    all_probs = []
    all_ids = []
    
    model.eval()
    with torch.no_grad():
        for aug_images, ids in tqdm(dataloader, desc='TTA Prediction'):
            # aug_images shape: [batch_size, num_augmentations, channels, height, width]
            batch_size, num_augs, c, h, w = aug_images.shape
            
            # Reshape to process all augmentations at once
            aug_images = aug_images.view(batch_size * num_augs, c, h, w)
            aug_images = aug_images.to(device)
            
            # Get predictions
            outputs = model(aug_images)
            probs = torch.softmax(outputs, dim=1)
            
            # Reshape back and average across augmentations
            probs = probs.view(batch_size, num_augs, -1)
            avg_probs = probs.mean(dim=1)  # Average across augmentations
            
            all_probs.append(avg_probs.cpu().numpy())
            all_ids.extend(ids)
    
    return np.vstack(all_probs), all_ids

def load_model(model_class, model_path, device):
    """Load a trained model"""
    model = model_class(num_classes=Config.NUM_CLASSES).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def main():
    print("=" * 70)
    print("DAY 5: TEST-TIME AUGMENTATION (TTA)")
    print("=" * 70)
    print(f"\nDevice: {Config.DEVICE}")
    print(f"TTA augmentations: {Config.TTA_TRANSFORMS}")
    
    # Choose which model to use for TTA
    available_models = []
    
    if os.path.exists('day2_best_model.pth'):
        available_models.append({
            'name': 'day2_best_model.pth (ResNet18)',
            'class': ResNet18Classifier,
            'path': 'day2_best_model.pth',
            'image_size': 224
        })
    
    if os.path.exists('day3_best_model.pth'):
        available_models.append({
            'name': 'day3_best_model.pth (EfficientNet-B1)',
            'class': EfficientNetB1Classifier,
            'path': 'day3_best_model.pth',
            'image_size': 256
        })
    
    if not available_models:
        print("\n✗ No trained models found!")
        print("Train a model first (Day 2 or Day 3)")
        return
    
    print("\nAvailable models:")
    for i, model_info in enumerate(available_models):
        print(f"  {i+1}. {model_info['name']}")
    
    # Use the best model (usually the last one trained)
    selected_model = available_models[-1]
    print(f"\n✓ Using: {selected_model['name']}")
    
    # Create TTA transforms
    print(f"\nCreating {Config.TTA_TRANSFORMS} TTA transforms...")
    tta_transforms = get_tta_transforms(image_size=selected_model['image_size'])
    
    # Create TTA dataset
    print("Creating TTA dataset...")
    tta_dataset = TTADataset(Config.TEST_DIR, tta_transforms)
    tta_loader = DataLoader(
        tta_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=2,  # Lower due to multiple augmentations
        pin_memory=True
    )
    
    print(f"Total test images: {len(tta_dataset)}")
    print(f"Each image will be augmented {Config.TTA_TRANSFORMS} times")
    print(f"Total predictions: {len(tta_dataset) * Config.TTA_TRANSFORMS}")
    
    # Load model
    print("\nLoading model...")
    model = load_model(selected_model['class'], selected_model['path'], Config.DEVICE)
    
    # Make predictions with TTA
    print("\nGenerating TTA predictions...")
    print("(This will take longer than normal prediction)")
    tta_probs, image_ids = predict_with_tta(model, tta_loader, Config.DEVICE)
    
    # Convert to predictions
    tta_preds = tta_probs.argmax(axis=1)
    
    # Show prediction distribution
    print("\n" + "=" * 70)
    print("TTA PREDICTION DISTRIBUTION")
    print("=" * 70)
    for i, class_name in enumerate(Config.CLASSES):
        count = (tta_preds == i).sum()
        pct = (count / len(tta_preds)) * 100
        print(f"  {class_name:12s}: {count:4d} ({pct:5.2f}%)")
    
    # Create submission
    formatted_ids = [str(img_id).zfill(5) for img_id in image_ids]
    submission = pd.DataFrame({
        'Id': formatted_ids,
        'Label': [Config.CLASSES[pred] for pred in tta_preds]
    })
    submission = submission.sort_values('Id').reset_index(drop=True)
    submission.to_csv('submission_day5_tta.csv', index=False, encoding='utf-8', lineterminator='\n')
    
    print("\n" + "=" * 70)
    print("✓ SUBMISSION CREATED")
    print("=" * 70)
    print("File: submission_day5_tta.csv")
    print(f"\nUsed model: {selected_model['name']}")
    print(f"TTA augmentations: {Config.TTA_TRANSFORMS}")
    print("\n🎯 Expected improvement: +0.2-0.8% over single prediction")
    print("🎯 Expected Kaggle score: 95.6-96.5%")
    
    # Compare with single prediction (without TTA)
    print("\n" + "=" * 70)
    print("TTA BENEFITS")
    print("=" * 70)
    print("\nTTA helps by:")
    print("  • Reducing prediction variance")
    print("  • Making model more robust to small changes")
    print("  • Averaging out random fluctuations")
    print("  • Improving confidence on borderline cases")
    print("\nBest used on:")
    print("  • Your best single model")
    print("  • Or combined with ensemble (Day 4)")

if __name__ == '__main__':
    main()
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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import shared modules
from common.config import Config
from common.models import get_model, ResNet18Classifier, EfficientNetB1Classifier

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

# TTA Transforms
def get_tta_transforms(image_size=256):
    """
    Create a list of TTA transforms using Albumentations.
    Each transform is a different augmentation that could help the model.
    """
    base_transform = [
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    
    tta_transforms = [
        # 1. Original image
        A.Compose(base_transform),
        
        # 2. Horizontal flip
        A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        
        # 3. Vertical flip
        A.Compose([
            A.Resize(image_size, image_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        
        # 4. Both flips
        A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        
        # 5. Slight rotation
        A.Compose([
            A.Resize(image_size, image_size),
            A.Rotate(limit=15, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        
        # 6. Brightness adjustment
        A.Compose([
            A.Resize(image_size, image_size),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        
        # 7. Scale
        A.Compose([
            A.Resize(image_size, image_size),
            A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.1, rotate_limit=0, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        
        # 8. Color jitter
        A.Compose([
            A.Resize(image_size, image_size),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    ]
    
    return tta_transforms

def predict_with_tta(model, dataloader, device):
    """
    Make predictions using Test-Time Augmentation.
    Returns averaged probabilities across all augmentations.
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

def load_trained_model(model_class, model_path, device):
    """Load a trained model checkpoint."""
    # Note: We need to know num_classes. Config has it.
    model = model_class(num_classes=Config.NUM_CLASSES).to(device)
    
    if not os.path.exists(model_path):
        # Check in output directory
        model_path = Config.OUTPUT_DIR / model_path
        
    if not os.path.exists(model_path):
         raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"Loading weights from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def main():
    print("=" * 70)
    print("DAY 5: TEST-TIME AUGMENTATION (TTA)")
    print("=" * 70)
    print(f"\nDevice: {Config.DEVICE}")
    
    # TTA settings
    TTA_COUNT = 8
    print(f"TTA augmentations: {TTA_COUNT}")
    
    # Search for available best models
    available_models = []
    
    # Check for models in current dir or outputs dir
    potential_models = [
        ('transfer_best_model.pth', ResNet18Classifier, 224),
        ('day2_best_model.pth', ResNet18Classifier, 224),
        ('day3_best_model.pth', EfficientNetB1Classifier, 224) 
    ]
    
    for filename, model_result_class, img_size in potential_models:
        # Check current dir
        if os.path.exists(filename):
             available_models.append({'name': filename, 'class': model_result_class, 'path': filename, 'size': img_size})
        # Check output dir
        elif (Config.OUTPUT_DIR / filename).exists():
             available_models.append({'name': filename, 'class': model_result_class, 'path': Config.OUTPUT_DIR / filename, 'size': img_size})

    if not available_models:
        print("\n✗ No trained models found!")
        print("Train a model first (e.g., run train_transfer_learning.py)")
        return
    
    print("\nAvailable models:")
    for i, model_info in enumerate(available_models):
        print(f"  {i+1}. {model_info['name']}")
    
    # Use the last one found (usually the most recent)
    selected_model = available_models[0] # Pick the first one for now or let user choose
    print(f"\n✓ Using: {selected_model['name']}")
    
    # Create TTA transforms
    print(f"\nCreating {TTA_COUNT} TTA transforms...")
    tta_transforms = get_tta_transforms(image_size=selected_model['size'])
    
    # Create TTA dataset
    print("Creating TTA dataset...")
    # Using TEST_DIR from Config
    tta_dataset = TTADataset(Config.TEST_DIR, tta_transforms)
    
    tta_loader = DataLoader(
        tta_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=2,  # Lower due to multiple augmentations amplification
        pin_memory=True
    )
    
    print(f"Total test images: {len(tta_dataset)}")
    
    # Load model
    print("\nLoading model...")
    model = load_trained_model(selected_model['class'], selected_model['path'], Config.DEVICE)
    
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
    # Convert numeric IDs to strings zfilled
    formatted_ids = [str(img_id).zfill(5) for img_id in image_ids]
    
    submission = pd.DataFrame({
        'Id': formatted_ids,
        'Label': [Config.CLASSES[pred] for pred in tta_preds]
    })
    
    submission = submission.sort_values('Id').reset_index(drop=True)
    
    out_path = Config.OUTPUT_DIR / 'submission_tta.csv'
    submission.to_csv(out_path, index=False)
    
    print("\n" + "=" * 70)
    print("✓ SUBMISSION CREATED")
    print("=" * 70)
    print(f"File: {out_path}")
    print(f"\nUsed model: {selected_model['name']}")

if __name__ == '__main__':
    main()
"""
Day 1 - Baseline CNN Model for Emoji Classification
Simple CNN trained from scratch - establishing baseline performance
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import shared modules
from common.config import Config
from common.dataset import EmojiDataset
from common.utils import set_seed

# Set random seeds
set_seed(Config.SEED)

# Simple CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation', leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# Main training loop
def main():
    print(f"Using device: {Config.DEVICE}")
    print(f"Number of classes: {Config.NUM_CLASSES}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE_BASELINE, Config.IMAGE_SIZE_BASELINE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load training data - preserve Id as string to maintain zero-padding
    train_labels = pd.read_csv(Config.TRAIN_LABELS, dtype={'Id': str})
    
    # Split into train and validation
    train_df, val_df = train_test_split(
        train_labels, 
        test_size=Config.VAL_SPLIT, 
        random_state=Config.SEED, 
        stratify=train_labels['Label']
    )
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Create datasets
    train_dataset = EmojiDataset(Config.TRAIN_DIR, train_df, transform=train_transform)
    val_dataset = EmojiDataset(Config.TRAIN_DIR, val_df, transform=train_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4
    )
    
    # Initialize model
    model = SimpleCNN(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5
    )
    
    # Training loop
    best_acc = 0.0
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, Config.DEVICE)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), Config.OUTPUT_DIR / 'baseline_best_model.pth')
            print(f"✓ Saved best model with accuracy: {best_acc:.2f}%")
    
    print(f"\nTraining completed! Best validation accuracy: {best_acc:.2f}%")
    
    # Generate predictions
    print("\nGenerating test predictions...")
    model.load_state_dict(torch.load(Config.OUTPUT_DIR / 'baseline_best_model.pth'))
    model.eval()
    
    test_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE_BASELINE, Config.IMAGE_SIZE_BASELINE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = EmojiDataset(Config.TEST_DIR, labels_df=None, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4
    )
    
    predictions = []
    image_ids = []
    
    with torch.no_grad():
        for images, ids in tqdm(test_loader, desc='Testing'):
            images = images.to(Config.DEVICE)
            outputs = model(images)
            _, preds = outputs.max(1)
            
            predictions.extend(preds.cpu().numpy())
            image_ids.extend(ids)
    
    # Create submission file
    submission = pd.DataFrame({
        'Id': image_ids,
        'Label': [Config.CLASSES[p] for p in predictions]
    })
    
    # Sort and save
    submission['Id'] = submission['Id'].astype(str).str.zfill(5)
    submission = submission.sort_values('Id')
    
    submission_path = Config.OUTPUT_DIR / 'submission_baseline.csv'
    submission.to_csv(submission_path, index=False)
    print(f"\n✓ Submission file saved to {submission_path}")
    print(f"Total predictions: {len(submission)}")

if __name__ == '__main__':
    main()
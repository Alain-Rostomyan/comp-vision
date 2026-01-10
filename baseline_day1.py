"""
Day 1 - Baseline CNN Model for Emoji Classification
Simple CNN trained from scratch - establishing baseline performance
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configuration
class Config:
    # Paths - MODIFY THESE TO MATCH YOUR DIRECTORY STRUCTURE
    # Get the directory where this script is located
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_DIR = os.path.join(BASE_DIR, 'train')
    TEST_DIR = os.path.join(BASE_DIR, 'test')
    TRAIN_LABELS = os.path.join(BASE_DIR, 'train_labels.csv')
    SAMPLE_SUBMISSION = os.path.join(BASE_DIR, 'sample_submission.csv')
    
    # Model parameters
    IMAGE_SIZE = 128
    BATCH_SIZE = 64
    EPOCHS = 20
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Classes
    CLASSES = ["apple", "facebook", "google", "messenger", "mozilla", "samsung", "whatsapp"]
    NUM_CLASSES = len(CLASSES)
    
    # Seed for reproducibility
    SEED = 42

# Set random seeds
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
set_seed(Config.SEED)

# Custom Dataset
class EmojiDataset(Dataset):
    def __init__(self, image_dir, labels_df=None, transform=None):
        self.image_dir = image_dir
        self.labels_df = labels_df
        self.transform = transform
        
        if labels_df is not None:
            self.image_ids = labels_df['Id'].values
            self.labels = labels_df['Label'].values
        else:
            # For test set
            self.image_ids = sorted([f.split('.')[0] for f in os.listdir(image_dir) if f.endswith('.png')])
            self.labels = None
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        # Ensure ID is zero-padded to 5 digits (e.g., 2228 -> 02228)
        img_id_str = str(img_id).zfill(5)
        img_path = os.path.join(self.image_dir, f"{img_id_str}.png")
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.labels is not None:
            label = Config.CLASSES.index(self.labels[idx])
            return image, label
        else:
            return image, img_id

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
    
    for images, labels in tqdm(dataloader, desc='Training'):
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
        for images, labels in tqdm(dataloader, desc='Validation'):
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
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load training data - preserve Id as string to maintain zero-padding
    train_labels = pd.read_csv(Config.TRAIN_LABELS, dtype={'Id': str})
    
    # Split into train and validation
    train_df, val_df = train_test_split(train_labels, test_size=0.15, random_state=Config.SEED, stratify=train_labels['Label'])
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Create datasets
    train_dataset = EmojiDataset(Config.TRAIN_DIR, train_df, transform=train_transform)
    val_dataset = EmojiDataset(Config.TRAIN_DIR, val_df, transform=train_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Initialize model
    model = SimpleCNN(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
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
            torch.save(model.state_dict(), 'day1_best_model.pth')
            print(f"✓ Saved best model with accuracy: {best_acc:.2f}%")
    
    print(f"\nTraining completed! Best validation accuracy: {best_acc:.2f}%")
    
    # Generate predictions
    print("\nGenerating test predictions...")
    model.load_state_dict(torch.load('day1_best_model.pth'))
    model.eval()
    
    test_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = EmojiDataset(Config.TEST_DIR, labels_df=None, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    
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
        'Label': [predictions[i] for i in range(len(predictions))]
    })
    
    submission.to_csv('submission_day1.csv', index=False)
    print("\n✓ Submission file saved as 'submission_day1.csv'")
    print(f"Total predictions: {len(submission)}")
    print("\nClass distribution in predictions:")
    print(submission['Label'].value_counts().sort_index())

if __name__ == '__main__':
    main()
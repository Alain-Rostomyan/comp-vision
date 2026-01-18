"""
Day 2 - Transfer Learning with ResNet18
Using pretrained ResNet18 from ImageNet for better feature extraction
Expected improvement: 65-75% accuracy
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import shared modules
from common.config import Config
from common.dataset import EmojiDataset
from common.utils import set_seed, save_checkpoint

# Set random seeds
set_seed(Config.SEED)

# ResNet18 Model
class EmojiClassifier(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super(EmojiClassifier, self).__init__()
        
        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=pretrained)
        
        # Replace final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

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
    print("=" * 60)
    print("DAY 2: Transfer Learning with ResNet18")
    print("=" * 60)
    print(f"\nUsing device: {Config.DEVICE}")
    print(f"Image size: {Config.IMAGE_SIZE_RESNET}x{Config.IMAGE_SIZE_RESNET}")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Epochs: {Config.EPOCHS}")
    
    # Data transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE_RESNET, Config.IMAGE_SIZE_RESNET)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE_RESNET, Config.IMAGE_SIZE_RESNET)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load training data
    print("\nLoading training data...")
    train_labels = pd.read_csv(Config.TRAIN_LABELS, dtype={'Id': str})
    
    # Split into train and validation (stratified)
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
    val_dataset = EmojiDataset(Config.TRAIN_DIR, val_df, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    print("\nInitializing ResNet18 model...")
    model = EmojiClassifier(num_classes=Config.NUM_CLASSES, pretrained=True).to(Config.DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        patience=3, 
        factor=0.5
    )
    
    # Training loop
    best_acc = 0.0
    patience_counter = 0
    
    print("\nStarting training...")
    print("=" * 60)
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        print("-" * 60)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, Config.DEVICE)
        
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Learning rate scheduling
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.6f}")
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, 'transfer_best_model.pth')
            print(f"  ✓ Saved best model! Val Acc: {best_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}")
        
        # Early stopping
        if patience_counter >= Config.PATIENCE:
            print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
            break
    
    print("\n" + "=" * 60)
    print(f"Training completed! Best validation accuracy: {best_acc:.2f}%")
    print("=" * 60)
    
    # Generate predictions
    print("\nGenerating test predictions...")
    
    # Load best model
    checkpoint = torch.load(Config.OUTPUT_DIR / 'transfer_best_model.pth', map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE_RESNET, Config.IMAGE_SIZE_RESNET)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = EmojiDataset(Config.TEST_DIR, labels_df=None, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
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
        'Id': [str(img_id) for img_id in image_ids],
        'Label': [Config.CLASSES[p] for p in predictions]
    })
    
    # Ensure proper sorting by ID
    submission['Id'] = submission['Id'].astype(str).str.zfill(5)
    submission = submission.sort_values('Id')
    
    submission_path = Config.OUTPUT_DIR / 'submission_transfer.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"\n✓ Submission created at {submission_path}")
    print(f"Total predictions: {len(submission)}")

if __name__ == '__main__':
    main()
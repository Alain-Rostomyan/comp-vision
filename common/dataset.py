"""
Dataset class for loading Emoji images.
"""
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple, List, Union
import pandas as pd
from common.config import Config

class EmojiDataset(Dataset):
    """
    Custom Dataset for Emoji Classification.
    """
    def __init__(
        self, 
        image_dir: Union[str, os.PathLike], 
        labels_df: Optional[pd.DataFrame] = None, 
        transform: Optional[Callable] = None
    ):
        """
        Args:
            image_dir: Directory with all the images.
            labels_df: DataFrame containing 'Id' and 'Label' (for training). 
                       If None, assumes test mode.
            transform: pytorch transforms for augmentation/normalization.
        """
        self.image_dir = image_dir
        self.labels_df = labels_df
        self.transform = transform
        
        if labels_df is not None:
            # Training mode: Use IDs from the dataframe
            self.image_ids = labels_df['Id'].values
            self.labels = labels_df['Label'].values
        else:
            # Test mode: load all images from directory
            self.image_ids = sorted([
                f.split('.')[0] 
                for f in os.listdir(image_dir) 
                if f.endswith('.png')
            ])
            self.labels = None
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[int, str]]:
        img_id = self.image_ids[idx]
        
        # Determine image path. Handle both string IDs and int IDs if necessary.
        # Training CSV often has IDs as int, but filenames are zero-padded string? 
        # Let's standardize on string for filename lookup.
        img_id_str = str(img_id).zfill(5)
        img_path = os.path.join(self.image_dir, f"{img_id_str}.png")
        
        # Load image
        # Convert to RGB to ensure 3 channels (handles RGBA or Greyscale)
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.labels is not None:
            # Return image and its label index
            # Label in DataFrame is usually the string name (e.g., 'apple')
            # changing to index based on Config.CLASSES
            label_name = self.labels[idx]
            label_idx = Config.CLASSES.index(label_name)
            return image, label_idx
        else:
            # Test mode: return image and its ID (for submission)
            return image, img_id

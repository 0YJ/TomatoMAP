#!/usr/bin/env python3
"""
Classification dataset for TomatoMAP
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class BBCHDataset(Dataset):
    """Dataset class for BBCH classification."""
    
    def __init__(self, data_dir, split='train', transform=None):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root directory of dataset
            split: Dataset split ('train', 'val', 'test')
            transform: Image transformations
        """
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")
        
        # Get all classes
        self.classes = sorted([d for d in os.listdir(self.data_dir)
                             if os.path.isdir(os.path.join(self.data_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all samples
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
        
        print(f"Loaded {split} dataset: {len(self.samples)} images, {len(self.classes)} classes")
    
    def __len__(self):
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get item by index."""
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Failed to load image: {img_path}, error: {e}")
            # Return black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(target_size=(640, 640)):
    """
    Get data transformations for training and validation.
    
    Args:
        target_size: Target image size (width, height)
    
    Returns:
        train_transform, val_transform
    """
    train_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

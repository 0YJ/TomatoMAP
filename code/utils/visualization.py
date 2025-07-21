#!/usr/bin/env python3
"""
Visualization utilities for TomatoMAP project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path
import torch


def plot_training_curves(history, output_dir):
    """
    Plot training curves for loss, accuracy, and learning rate.
    
    Args:
        history: Dictionary containing training history
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='red')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Acc', color='blue')
    plt.plot(history['val_acc'], label='Val Acc', color='red')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot learning rate
    plt.subplot(1, 3, 3)
    plt.plot(history['lr'], label='Learning Rate', color='green')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(output_dir) / 'training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {save_path}")


def create_confusion_matrix(true_labels, pred_labels, class_names, output_dir):
    """
    Create and save confusion matrix visualization.
    
    Args:
        true_labels: True labels
        pred_labels: Predicted labels
        class_names: List of class names
        output_dir: Directory to save outputs
    """
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Save raw confusion matrix
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(Path(output_dir) / 'confusion_matrix.csv')
    
    # Normalize confusion matrix
    normalized_cm = cm_df.div(cm_df.sum(axis=1), axis=0).fillna(0)
    matrix = normalized_cm.T.to_numpy()
    
    # Create visualization
    from matplotlib.colors import Normalize
    masked_matrix = np.ma.masked_where(matrix == 0, matrix)
    
    cmap = plt.cm.jet
    cmap.set_bad(color='white')
    norm = Normalize(vmin=0.1, vmax=1)
    
    fig_size = 3.1
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    im = ax.imshow(masked_matrix, cmap=cmap, norm=norm)
    
    # Remove ticks and labels for publication
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    plt.tight_layout()
    save_path = Path(output_dir) / 'normalized_confusion_matrix.png'
    plt.savefig(save_path, format='png', dpi=300)
    plt.close()
    
    print(f"Confusion matrix saved to: {save_path}")


def denormalize(img_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Denormalize image tensor for visualization.
    
    Args:
        img_tensor: Normalized image tensor
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Denormalized image tensor
    """
    mean = torch.tensor(mean).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(img_tensor.device)
    return torch.clamp(img_tensor * std + mean, 0, 1)

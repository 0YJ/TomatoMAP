#!/usr/bin/env python3
"""
Visualization Utilities for TomatoMAP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path
import torch


def plot_training_curves(history, output_dir):

    plt.figure(figsize=(15, 5))
    
    # plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='red')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Acc', color='blue')
    plt.plot(history['val_acc'], label='Val Acc', color='red')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # plot learning rate
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

    # calculate confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # save raw confusion matrix
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(Path(output_dir) / 'confusion_matrix.csv')
    
    # normalize confusion matrix
    normalized_cm = cm_df.div(cm_df.sum(axis=1), axis=0).fillna(0)
    matrix = normalized_cm.T.to_numpy()
    
    # create visualization
    from matplotlib.colors import Normalize
    masked_matrix = np.ma.masked_where(matrix == 0, matrix)
    
    cmap = plt.cm.jet
    cmap.set_bad(color='white')
    norm = Normalize(vmin=0.1, vmax=1)
    
    fig_size = 3.1
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    im = ax.imshow(masked_matrix, cmap=cmap, norm=norm)
    
    # remove ticks and labels for publishing
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

    mean = torch.tensor(mean).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(img_tensor.device)
    return torch.clamp(img_tensor * std + mean, 0, 1)

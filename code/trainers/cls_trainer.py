#!/usr/bin/env python3
"""
Classification trainer for TomatoMAP dataset
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from datasets.cls_dataset import BBCHDataset, get_transforms
from models.cls_models import get_model
from utils.common import (
    save_model, load_model, save_checkpoint, load_checkpoint,
    create_output_dir, print_config, print_section, format_time,
    get_num_workers
)
from utils.visualization import plot_training_curves, create_confusion_matrix


class ClassificationTrainer:
    """Trainer class for classification models."""
    
    def __init__(self, args):
        """Initialize trainer with arguments."""
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set output directory
        if args.output_dir is None:
            self.output_dir = create_output_dir('outputs/cls', f'{args.model}_cls')
        else:
            self.output_dir = create_output_dir(args.output_dir)
        
        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_training()
        
    def _setup_data(self):
        """Setup datasets and dataloaders."""
        print_section("Setting up data")
        
        # Get transforms
        self.train_transform, self.val_transform = get_transforms(tuple(self.args.img_size))
        
        # Create datasets
        self.train_dataset = BBCHDataset(
            self.args.data_dir, 'train', self.train_transform
        )
        self.val_dataset = BBCHDataset(
            self.args.data_dir, 'val', self.val_transform
        )
        
        # Create dataloaders
        num_workers = get_num_workers()
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        # Try to load test dataset
        test_dir = os.path.join(self.args.data_dir, 'test')
        if os.path.exists(test_dir):
            self.test_dataset = BBCHDataset(
                self.args.data_dir, 'test', self.val_transform
            )
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
        else:
            print("Test set not found, will use validation set for testing")
            self.test_loader = self.val_loader
        
    def _setup_model(self):
        """Setup model architecture."""
        print_section("Setting up model")
        
        self.model = get_model(
            self.args.model,
            self.args.num_classes,
            pretrained=True
        )
        self.model = self.model.to(self.device)
        
    def _setup_training(self):
        """Setup training components."""
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=0.01
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=30,
            gamma=0.1
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.start_epoch = 0
        
        # Resume from checkpoint if specified
        if self.args.resume:
            self.start_epoch, checkpoint = load_checkpoint(
                self.args.resume,
                self.model,
                self.optimizer,
                self.device
            )
            if 'best_val_acc' in checkpoint:
                self.best_val_acc = checkpoint['best_val_acc']
            if 'history' in checkpoint:
                self.history = checkpoint['history']
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.args.epochs} [Train]"
        )
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            current_acc = 100 * train_correct / train_total
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        avg_loss = train_loss / len(self.train_loader)
        accuracy = 100 * train_correct / train_total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, epoch):
        """Validate for one epoch."""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(
                self.val_loader,
                desc=f"Epoch {epoch+1}/{self.args.epochs} [Val]"
            )
            
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                current_acc = 100 * val_correct / val_total
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
        
        avg_loss = val_loss / len(self.val_loader)
        accuracy = 100 * val_correct / val_total
        
        return avg_loss, accuracy
    
    def train(self):
        """Main training loop."""
        # Print configuration
        config = {
            'Data directory': self.args.data_dir,
            'Model': self.args.model,
            'Number of classes': self.args.num_classes,
            'Batch size': self.args.batch_size,
            'Epochs': self.args.epochs,
            'Learning rate': self.args.lr,
            'Image size': self.args.img_size,
            'Patience': self.args.patience,
            'Device': self.device,
            'Output directory': self.output_dir
        }
        print_config(config, "Training Configuration")
        
        print_section("Starting training")
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(epoch)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.args.epochs}:")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"  LR: {current_lr:.2e}")
            
            # Check for improvement
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                save_model(self.model, self.output_dir / f'best_{self.args.model}.pth')
                self.patience_counter = 0
                print(f"  New best validation accuracy: {self.best_val_acc:.2f}%")
            else:
                self.patience_counter += 1
                print(f"  No improvement ({self.patience_counter}/{self.args.patience})")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch+1}.pth'
                save_checkpoint(
                    self.model, self.optimizer, epoch, checkpoint_path,
                    best_val_acc=self.best_val_acc,
                    history=self.history
                )
            
            # Early stopping
            if self.patience_counter >= self.args.patience:
                print(f"\nEarly stopping triggered after {self.args.patience} epochs without improvement")
                break
            
            print("-" * 60)
        
        # Save final model
        save_model(self.model, self.output_dir / f'final_{self.args.model}.pth')
        
        # Training summary
        total_time = time.time() - start_time
        print_section("Training completed")
        print(f"Total training time: {format_time(total_time)}")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Models saved in: {self.output_dir}")
        
        # Save training history
        history_file = self.output_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Plot training curves
        plot_training_curves(self.history, self.output_dir)
        
        # Evaluate on test set
        self.evaluate()
    
    def evaluate(self):
        """Evaluate model on test set."""
        print_section("Evaluating on test set")
        
        # Load best model
        best_model_path = self.output_dir / f'best_{self.args.model}.pth'
        if best_model_path.exists():
            load_model(self.model, best_model_path, self.device)
        
        self.model.eval()
        test_correct = 0
        test_total = 0
        test_predictions = []
        test_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing")
            
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                test_predictions.extend(predicted.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
                
                current_acc = 100 * test_correct / test_total
                pbar.set_postfix({'Acc': f'{current_acc:.2f}%'})
        
        test_accuracy = 100 * test_correct / test_total
        print(f"Test accuracy: {test_accuracy:.2f}%")
        
        # Create confusion matrix
        class_names = self.train_dataset.classes
        create_confusion_matrix(
            test_labels,
            test_predictions,
            class_names,
            self.output_dir
        )
        
        # Save test results
        test_results = {
            'test_accuracy': test_accuracy,
            'total_samples': test_total,
            'correct_predictions': test_correct,
            'num_classes': len(class_names),
            'class_names': class_names
        }
        
        results_file = self.output_dir / 'test_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        
        print_section("Evaluation Summary")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Test accuracy: {test_accuracy:.2f}%")
        print(f"Number of classes: {len(class_names)}")
        print(f"Test dataset size: {test_total}")
        print(f"Results saved in: {self.output_dir}")

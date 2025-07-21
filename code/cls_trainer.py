#!/usr/bin/env python3
"""
TomatoMAP-Cls Trainer
Classification trainer for TomatoMAP dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import ImageDraw, ImageFont, Image
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    MobileNet_V2_Weights,
    ResNet18_Weights,
)

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"model saved at: {path}")

def load_model(model, path, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"model loaded from: {path}")

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)

def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"from epoch {start_epoch} re-training")
    return start_epoch

def get_font(size=30, bold=False):
    font_paths = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial-Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size=size)
        except:
            continue
    return ImageFont.load_default()

def denormalize(img_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.tensor(mean).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(img_tensor.device)
    return torch.clamp(img_tensor * std + mean, 0, 1)

def get_model(name, num_classes, pretrained=True):
    print(f"build model: {name}, class number: {num_classes}")
    
    if name == 'mobilenet_v3_large':
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif name == 'mobilenet_v3_small':
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif name == 'mobilenet_v2':
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == 'resnet18':
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Model {name} not supported.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameter info: Total: {total_params:,}, Trainable: {trainable_params:,}")
    
    return model

class BBCHDataset(Dataset):
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")
        
        # get all classes
        self.classes = sorted([d for d in os.listdir(self.data_dir)
                              if os.path.isdir(os.path.join(self.data_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
        
        print(f"loading {split} dataset: {len(self.samples)} images, {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"failed to load image: {img_path}, error: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# data enhance
def get_transforms(target_size=(640, 640)):
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

def get_dataloaders(data_dir, batch_size=32, target_size=(640, 640), num_workers=8, include_test=False):
    print(f"building dataloader: {data_dir}")
    
    train_transform, val_transform = get_transforms(target_size)
    
    train_dataset = BBCHDataset(data_dir, 'train', train_transform)
    val_dataset = BBCHDataset(data_dir, 'val', val_transform)

    # for windows users
    import platform
    if platform.system() == 'Windows':
        num_workers = 0
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    test_loader = None
    if include_test:
        test_dir = os.path.join(data_dir, 'test')
        if os.path.exists(test_dir):
            test_dataset = BBCHDataset(data_dir, 'test', val_transform)
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=torch.cuda.is_available()
            )
        else:
            print("test set not found, using val as test")
            test_loader = val_loader

    return train_loader, val_loader, test_loader

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(f"cls/runs/{config['model_name']}_cls")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_loader, val_loader, test_loader = get_dataloaders(
        config['data_dir'], 
        batch_size=config['batch_size'],
        target_size=config['target_size'],
        num_workers=8,
        include_test=True
    )
    
    model = get_model(config['model_name'], config['num_classes'], pretrained=True)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"Training start with {config['num_epochs']} epoch(s),")
    
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
        
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            current_acc = 100 * train_correct / train_total
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Val]")
            
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                current_acc = 100 * val_correct / val_total
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n Epoch {epoch+1}/{config['num_epochs']}:")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"  lr: {current_lr:.2e}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = output_dir / f"best_{config['model_name']}.pth"
            save_model(model, best_model_path)
            patience_counter = 0
            print(f"  best val acc: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"  val acc not raised ({patience_counter}/{config['patience']})")
        
        if (epoch + 1) % config['save_interval'] == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
        
        if patience_counter >= config['patience']:
            print(f"\n Trigger early stop. Val acc has {config['patience']} epochs no improve")
            break
        
        print("-" * 60)
    
    final_model_path = output_dir / f"final_{config['model_name']}.pth"
    save_model(model, final_model_path)
    
    print(f"\n TomatoMAP-Cls is trained!")
    print(f"  best val acc: {best_val_acc:.2f}%")
    print(f"  model saved at: {output_dir}")
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='train loss', color='blue')
    plt.plot(val_losses, label='val loss', color='red')
    plt.title('training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='train acc', color='blue')
    plt.plot(val_accuracies, label='val acc', color='red')
    plt.title('training acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    lrs = []
    for i in range(len(train_losses)):
        if i < 30:
            lrs.append(config['learning_rate'])
        elif i < 60:
            lrs.append(config['learning_rate'] * 0.1)
        else:
            lrs.append(config['learning_rate'] * 0.01)
    plt.plot(lrs, label='lr', color='green')
    plt.title('lr changes')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accuracies,
        'val_acc': val_accuracies
    })
    history_df.to_csv(output_dir / 'training_history.csv', index=False)
    print(f" training log saved at: {output_dir / 'training_history.csv'}")
    
    return model, best_val_acc, output_dir, test_loader

def main():
    # env checker
    print("Environment checker:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA version: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU device: {torch.cuda.get_device_name(0)}")
        print(f"  GPU ram: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    CLASSIFICATION_CONFIG = {
        'data_dir': 'TomatoMAP/TomatoMAP-Cls',
        'model_name': 'mobilenet_v3_large',  # 'mobilenet_v3_small', 'mobilenet_v2', 'resnet18'
        'num_classes': 50,
        'batch_size': 32,
        'num_epochs': 30,
        'learning_rate': 1e-4,
        'target_size': (640, 640),
        'patience': 3,
        'save_interval': 20
    }

    print("config:")
    for key, value in CLASSIFICATION_CONFIG.items():
        print(f"  {key}: {value}")

    print("=" * 60)
    print("TomatoMAP-Cls Trainer")
    print("=" * 60)

    if not os.path.exists(CLASSIFICATION_CONFIG['data_dir']):
        print(f"dataset not exist")
        print(f"   path: {CLASSIFICATION_CONFIG['data_dir']}")
        print(f"   please check data structure")
    else:
        print(f"data founded at: {CLASSIFICATION_CONFIG['data_dir']}")
        
        train_dir = os.path.join(CLASSIFICATION_CONFIG['data_dir'], 'train')
        val_dir = os.path.join(CLASSIFICATION_CONFIG['data_dir'], 'val')
        test_dir = os.path.join(CLASSIFICATION_CONFIG['data_dir'], 'test')
        
        if not os.path.exists(train_dir):
            print(f"training subset not exist: {train_dir}")
        elif not os.path.exists(val_dir):
            print(f"val subset not exist: {val_dir}")
        elif not os.path.exists(test_dir):
            print(f"test subset not exist: {test_dir}")
            print(f"   using val subset for test")
        else:
            print(f"TomatoMAP-Cls is well structured.")
            
            print("\n training config:")
            for key, value in CLASSIFICATION_CONFIG.items():
                print(f"   {key}: {value}")
            
            print("\n training start.")
            
            try:
                model, best_acc, output_dir, test_loader = train_model(CLASSIFICATION_CONFIG)
                
                print("\n" + "=" * 60)
                print("\n training finished!")
                print(f"   best val acc is: {best_acc:.2f}%")
                print(f"   model saved at: {output_dir}")
                
                print("\n evaluating on test subset...")
                model.eval()
                test_correct = 0
                test_total = 0
                test_predictions = []
                test_labels = []
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                with torch.no_grad():
                    test_pbar = tqdm(test_loader, desc="evaluating")
                    for images, labels in test_pbar:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        
                        test_total += labels.size(0)
                        test_correct += (predicted == labels).sum().item()

                        test_predictions.extend(predicted.cpu().numpy())
                        test_labels.extend(labels.cpu().numpy())
                        
                        current_acc = 100 * test_correct / test_total
                        test_pbar.set_postfix({'Acc': f'{current_acc:.2f}%'})
                
                test_accuracy = 100 * test_correct / test_total
                print(f" test subset acc: {test_accuracy:.2f}%")

                print("\n building confusion matrix")
                
                train_dataset = test_loader.dataset
                class_names = train_dataset.classes
                
                cm = confusion_matrix(test_labels, test_predictions)
                
                cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
                cm_df.to_csv(output_dir / 'confusion_matrix.csv')

                normalized_cm = cm_df.div(cm_df.sum(axis=1), axis=0).fillna(0)
                
                matrix = normalized_cm.T.to_numpy()
                
                from matplotlib import rcParams
                # rcParams['font.family'] = 'Calibri' # Ubuntu doesn't own this when training on ubuntu VM
                rcParams['font.size'] = 8
                
                masked_matrix = np.ma.masked_where(matrix == 0, matrix)
                
                from matplotlib.colors import Normalize
                cmap = plt.cm.jet
                cmap.set_bad(color='white')
                norm = Normalize(vmin=0.1, vmax=1)
                
                fig_width_in = 3.1
                fig_height_in = fig_width_in
                fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))
                
                im = ax.imshow(masked_matrix, cmap=cmap, norm=norm)

                # For further process for publishing purpose, labels are removed :)
                ax.set_xlabel("")
                ax.set_ylabel("")
                
                ax.set_xticks([])
                ax.set_yticks([])
                
                plt.tight_layout()
                plt.savefig(output_dir / 'normalized_confusion_matrix.png', format='png', dpi=300)
                plt.show()
                
                test_results = {
                    'test_accuracy': test_accuracy,
                    'total_samples': test_total,
                    'correct_predictions': test_correct,
                    'num_classes': len(class_names),
                    'class_names': class_names
                }
                
                import json
                with open(output_dir / 'test_results.json', 'w', encoding='utf-8') as f:
                    json.dump(test_results, f, indent=2, ensure_ascii=False)
                
                print("\n" + "=" * 60)
                print(" Evaluation results:")
                print(f"   best val acc: {best_acc:.2f}%")
                print(f"   test acc: {test_accuracy:.2f}%")
                print(f"   class number: {len(class_names)}")
                print(f"   test data size: {test_total}")
                print(f"   results saved at: {output_dir}")
                print(f"   GGWP!")
                print("=" * 60)
                
            except KeyboardInterrupt:
                print("\n training interruptted")
                
            except Exception as e:
                print(f"\n error during training:")
                print(f"   error info: {str(e)}")
                print("\nDetails:")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()

# Usage: 
#   python train.py --model_name mobilenet_v3_large --num_classes 50
#   python train.py --resume checkpoints/last_checkpoint.pth

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from models.classifiers import get_model
from datasets.custom_dataset import get_dataloaders
from utils import save_model

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
    print(f"🔁 Resumed from checkpoint at epoch {start_epoch}")
    return start_epoch

def plot_training_curve(log_path, model_name):
    if not os.path.exists(log_path):
        print("📭 No training log found to plot.")
        return
    df = pd.read_csv(log_path)
    plt.figure()
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.plot(df['epoch'], df['train_accuracy'], label='Train Acc')
    plt.plot(df['epoch'], df['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(f'Training & Validation Curve - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"checkpoints/{model_name}_curve.png")
    print(f"📈 Saved training curve to checkpoints/{model_name}_curve.png")

def evaluate(model, dataloader, criterion, device):
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    avg_loss = loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def train_model(data_dir, num_classes, model_name='mobilenet_v3_large',
                batch_size=32, num_epochs=150, learning_rate=1e-4, target_size=(640, 640),
                resume_path=None, save_interval=20, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_dataloaders(data_dir, batch_size, target_size)
    model = get_model(model_name, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    start_epoch = 0
    if resume_path and os.path.isfile(resume_path):
        start_epoch = load_checkpoint(resume_path, model, optimizer, device)

    os.makedirs("checkpoints", exist_ok=True)
    log_path = f"checkpoints/{model_name}_train_log.csv"
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write("epoch,train_loss,train_accuracy,val_loss,val_accuracy\n")

    print("\n🔥 Starting training...")

    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\n🔁 Epoch {epoch+1}/{num_epochs}")

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
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

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"✅ Epoch {epoch+1} Finished - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        with open(log_path, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}\n")

        save_checkpoint(model, optimizer, epoch, f"checkpoints/last_checkpoint.pth")
        if (epoch + 1) % save_interval == 0:
            checkpoint_name = f"checkpoints/{model_name}_epoch{epoch+1}.pth"
            save_checkpoint(model, optimizer, epoch, checkpoint_name)
            print(f"💾 Saved checkpoint: {checkpoint_name}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            save_model(model, f'checkpoints/{model_name}_best.pth')
            print("🌟 New best model saved!")
        else:
            early_stop_counter += 1
            print(f"⏸️ Early stopping counter: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print("🛑 Early stopping triggered!")
                break

    save_model(model, f'{model_name}_custom.pth')
    plot_training_curve(log_path, model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='Root dataset directory')
    parser.add_argument('--num_classes', type=int, default=50, help='Number of classes')
    parser.add_argument('--model_name', type=str, default='mobilenet_v3_large', help='Model name')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=640, help='Image size (square)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')

    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        num_classes=args.num_classes,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        target_size=(args.img_size, args.img_size),
        resume_path=args.resume,
        save_interval=20,
        patience=args.patience
    )

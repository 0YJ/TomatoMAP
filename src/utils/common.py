#!/usr/bin/env python3
"""
Common Utilities for TomatoMAP
"""

import os
import torch
import platform
from pathlib import Path
from PIL import ImageFont


def print_header(title):
    # print a formatted header
    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60 + "\n")


def print_section(title):
    # print a formatted section header
    print(f"\n{title}")
    print("-" * len(title))


def check_environment():
    # check and print environment information
    print_section("Environment Check")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print(f"Platform: {platform.system()}")
    print(f"Python version: {platform.python_version()}")


def create_output_dir(base_dir, task_name=None):
    # create output directory structure
    output_dir = Path(base_dir)
    if task_name:
        output_dir = output_dir / task_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_model(model, path):
    # save model state dict
    torch.save(model.state_dict(), path)
    print(f"Model saved at: {path}")


def load_model(model, path, device='cpu'):
    # load model state dict
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Model loaded from: {path}")
    return model


def save_checkpoint(model, optimizer, epoch, path, **kwargs):
    # save training checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    checkpoint.update(kwargs)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer, device):
    # load training checkpoint
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming from epoch {start_epoch}")
    return start_epoch, checkpoint


def get_num_workers():
    # get appropriate number of workers based on platform (windows make it 0)
    if platform.system() == 'Windows':
        return 0
    return min(8, os.cpu_count() or 1)


def get_font(size=30, bold=False):
    # get font (for publishing purpose)
    font_paths = [
        # Windows
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        # macOS
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size=size)
        except:
            continue
    
    return ImageFont.load_default()


def print_config(config_dict, title="Configuration"):
    # print configuration in a formatted way
    print_section(title)
    max_key_len = max(len(key) for key in config_dict.keys())
    
    for key, value in config_dict.items():
        if isinstance(value, (list, tuple)) and len(value) == 2:
            print(f"  {key:<{max_key_len}} : {value[0]} x {value[1]}")
        else:
            print(f"  {key:<{max_key_len}} : {value}")


def format_time(seconds):
    # format seconds to readable time
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

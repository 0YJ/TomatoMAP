# TomatoMAP Training System

A unified training system for TomatoMAP dataset supporting classification, detection, and segmentation tasks.

## Overview

TomatoMAP is a comprehensive dataset for tomato plant analysis, containing:
- **TomatoMAP-Cls**: Classification dataset with 50 BBCH growth stages
- **TomatoMAP-Det**: Object detection dataset for tomato detection
- **TomatoMAP-Seg**: Instance segmentation dataset with 10 categories

## Installation

### Requirements

```bash
# Basic requirements
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib tqdm pillow scikit-learn

# Task-specific requirements
# For detection:
pip install ultralytics

# For segmentation:
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
pip install opencv-python

# For ISAT to COCO conversion:
pip install pyyaml
```

### Project Structure

```
TomatoMAP/
├── main.py                 # Main entry point
├── README.md              # This file
├── requirements.txt       # Dependencies
│
├── trainers/              # Training modules
│   ├── cls_trainer.py     # Classification trainer
│   ├── det_trainer.py     # Detection trainer
│   └── seg_trainer.py     # Segmentation trainer
│
├── datasets/              # Dataset handling
│   ├── cls_dataset.py     # Classification dataset
│   └── seg_dataset.py     # Segmentation dataset utilities
│
├── models/                # Model definitions
│   ├── cls_models.py      # Classification models
│   └── seg_hooks.py       # Segmentation training hooks
│
├── utils/                 # Utility functions
│   ├── common.py          # Common utilities
│   ├── visualization.py   # Visualization tools
│   └── converters.py      # Format converters
│
└── outputs/               # Training outputs (created automatically)
    ├── cls/              # Classification results
    ├── det/              # Detection results
    └── seg/              # Segmentation results
```

## Usage

### Classification Training

Train a classification model on TomatoMAP-Cls dataset:

```bash
# Basic training with MobileNetV3-Large
python main.py cls --data-dir ./TomatoMAP/TomatoMAP-Cls --epochs 100

# Advanced options
python main.py cls \
    --data-dir ./TomatoMAP/TomatoMAP-Cls \
    --model mobilenet_v3_large \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4 \
    --img-size 640 640 \
    --patience 5 \
    --output-dir outputs/cls/experiment1
```

Available models:
- `mobilenet_v3_large` (default)
- `mobilenet_v3_small`
- `mobilenet_v2`
- `resnet18`

### Detection Training

Train a YOLO model on TomatoMAP-Det dataset:

```bash
# Basic training with YOLO11-Large
python main.py det --data-config ./det/TomatoMAP-Det.yaml --epochs 500

# Advanced options
python main.py det \
    --data-config ./det/TomatoMAP-Det.yaml \
    --model yolo11l.pt \
    --epochs 500 \
    --img-size 640 \
    --batch-size 4 \
    --patience 10 \
    --device 0 \
    --output-dir outputs/det/experiment1 \
    --hyperparams ./det/best_hyperparameters.yaml
```

### Segmentation Training

Train a Mask R-CNN model on TomatoMAP-Seg dataset:

```bash
# Training
python main.py seg train \
    --data-dir ./TomatoMAP/TomatoMAP-Seg \
    --model COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
    --epochs 100 \
    --lr 0.0001 \
    --batch-size 4 \
    --patience 5

# Evaluation
python main.py seg eval \
    --data-dir ./TomatoMAP/TomatoMAP-Seg \
    --model-path model_best.pth \
    --output-dir outputs/seg

# Visualization
python main.py seg vis \
    --data-dir ./TomatoMAP/TomatoMAP-Seg \
    --model-path model_best.pth \
    --n 5 \
    --output-dir outputs/seg

# Dataset information
python main.py seg info --data-dir ./TomatoMAP/TomatoMAP-Seg

# Analyze object areas
python main.py seg analyze --data-dir ./TomatoMAP/TomatoMAP-Seg
```

Available models:
- `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml`
- `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml`
- `COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml`
- `COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml`

## Dataset Preparation

### Classification Dataset Structure
```
TomatoMAP-Cls/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── ...
│   └── class2/
│       └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

### Detection Dataset Structure
```
det/
├── TomatoMAP-Det.yaml    # YOLO data configuration
└── best_hyperparameters.yaml  # Optional hyperparameters
```

### Segmentation Dataset Structure
```
TomatoMAP-Seg/
├── images/               # All images
│   ├── img1.jpg
│   └── ...
└── cocoOut/             # COCO format annotations

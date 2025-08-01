# ![logo](../imgs/logo.png)TomatoMAP Preprocessing and Training Space
***A Novel Dataset for Tomato Fine-Grained Phenotyping*** 

We offer a preprocessing pipeline and support training sample models for classification, detection, and segmentation tasks.

## Overview

TomatoMAP is a comprehensive dataset for tomato plant analysis, containing:
- **TomatoMAP-Cls**: Classification dataset with 50 BBCH growth stages
- **TomatoMAP-Det**: Object detection dataset for tomato detection
- **TomatoMAP-Seg**: Instance segmentation dataset with 10 categories

## Getting start
Download [TomatoMAP](https://doi.ipk-gatersleben.de/DOI/89386758-8bfd-41ca-aa9c-ee363e9d94c9/073051f0-b05e-4b43-a9cd-0435fe7cd913/2/1847940088), unzip it under code/ folder, and run through **TomatoMAP_builder.ipynb** to preprocess the dataset to TomatoMAP-Cls, TomatoMAP-Det. TomatoMAP-Seg is included directly by download. 

### Requirements
We suggest using [conda](https://www.anaconda.com/) for env management. 
```
conda create -n tomatomap python=3.10
conda activate tomatomap
```
We use notebook as TomatoMAP builder (script version coming soon).
```bash
pip install notebook
jupyter notebook
```
Clone repo.
```bash
# clone repo
git clone https://github.com/0YJ/TomatoMAP.git && cd TomatoMAP/code
cp det/best_hyperparameters.yaml ./

# install [PyTorch](https://pytorch.org/get-started/locally/)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install numpy pandas matplotlib tqdm pillow scikit-learn

# task-specific requirements
# for detection:
pip install ultralytics

# for segmentation:
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install opencv-python

# for ISAT2COCO conversion:
pip install pyyaml

# translate ISAT format to COCO
cd utils
python isat2coco.py
```

### Project Structure

```
TomatoMAP/
├── main.py                # Main entry
├── README.md              # Introduction
├── requirements.txt       # Dependencies
│
├── avh/                   # AI vs Human Analysis
│
├── seg/                   # Segmentation package
│
├── cls/                   # Classifier
│
├── det/                   # Detection package
│   ├── TomatoMAP-Det.yaml     # YOLo training settings
│   └── best_hyperparameters.yaml     # Fine-tuned hyperparameters
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
│   └── isat2coco.py       # Format converter for Seg
│
└── outputs/              # Training outputs (created automatically)
    ├── cls/              # Classification results
    ├── det/              # Detection results
    └── seg/              # Segmentation results
```

## Usage

### Classification Training

Train a classification model on TomatoMAP-Cls dataset:

```bash
# default training with MobileNetV3-Large
python main.py cls --data-dir ./TomatoMAP/TomatoMAP-Cls --epochs 100

# options
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
# default training with YOLO11-Large
python main.py det --data-config ./det/TomatoMAP-Det.yaml --epochs 500

# options
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

Train a Mask R-CNN FPN based model on TomatoMAP-Seg dataset:

```bash
# training
python main.py seg train \
    --data-dir ./TomatoMAP/TomatoMAP-Seg \
    --model COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
    --epochs 100 \
    --lr 0.0001 \
    --batch-size 4 \
    --patience 5

# evaluation
python main.py seg eval \
    --data-dir ./TomatoMAP/TomatoMAP-Seg \
    --model-path model_best.pth \
    --output-dir outputs/seg

# visualization
python main.py seg vis \
    --data-dir ./TomatoMAP/TomatoMAP-Seg \
    --model-path model_best.pth \
    --n 5 \
    --output-dir outputs/seg

# dataset information
python main.py seg info --data-dir ./TomatoMAP/TomatoMAP-Seg

# analyze object size (small, big, middle)
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
│   ├── BBCH class1/
│   │   ├── img1.jpg
│   │   └── ...
│   └── BBCH class2/
│       └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

### Detection Dataset Structure
```
TomatoMAP-Det/
├── images
└── labels
```

### Segmentation Dataset Structure
```
TomatoMAP-Seg/
├── images/               # All images
│   ├── img1.JPG
│   └── ...
├── labels/               # All labels in COCO format
    ├── isat.yaml         # Label and class configuration
    └── img1.json
```

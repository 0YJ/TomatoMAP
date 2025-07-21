## Usage

### Quick Start

Check your environment:
```bash
python main.py --check-env
```

Train with default settings:
```bash
python main.py cls  # Classification
python main.py det  # Detection  
python main.py seg  # Segmentation
python main.py all  # All models
```

### Advanced Usage with Custom Parameters

```bash
# Classification with custom parameters
python main.py cls --cls-model resnet18 --cls-epochs 50 --cls-lr 0.001 --cls-batch 16

# Detection with custom parameters  
python main.py det --det-epochs 300 --det-batch 8 --det-patience 15

# Segmentation with custom parameters
python main.py seg --seg-model COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml --seg-lr 0.0005 --seg-epochs 150

# Train all with evaluation and visualization
python main.py all --eval --vis --vis-samples 5 --output ./results
```

### Parameter Reference

**Classification Parameters:**
- `--cls-model`: Model architecture (mobilenet_v3_large, mobilenet_v3_small, mobilenet_v2, resnet18)
- `--cls-epochs`: Training epochs (default: 30)
- `--cls-lr`: Learning rate (default: 1e-4)
- `--cls-batch`: Batch size (default: 32)
- `--cls-patience`: Early stopping patience (default: 3)
- `--cls-data`: Data directory (default: TomatoMAP/TomatoMAP-Cls)

**Detection Parameters:**
- `--det-model`: YOLO model file (default: yolo11l.pt)
- `--det-epochs`: Training epochs (default: 500)
- `--det-batch`: Batch size (default: 4)
- `--det-patience`: Early stopping patience (default: 10)
- `--det-data`: Dataset YAML file (default: det/TomatoMAP-Det.yaml)
- `--det-cfg`: Hyperparameters config (default: det/best_hyperparameters.yaml)

**Segmentation Parameters:**
- `--seg-model`: Model config (default: mask_rcnn_R_50_FPN_1x.yaml)
- `--seg-epochs`: Training epochs (default: 100)
- `--seg-lr`: Learning rate (default: 0.0001)
- `--seg-batch`: Batch size (default: 4)
- `--seg-patience`: Early stopping patience (default: 5)
- `--seg-classes`: Number of classes (default: 10)

**Common Parameters:**
- `--output`: Output directory for all results
- `--eval`: Run evaluation after training
- `--vis`: Generate visualizations
- `--vis-samples`: Number of visualization samples (default: 3)

### Individual Training Scripts

You can still run individual training scripts with their specific arguments:

```bash
# Classification
python train_classifier.py

# Detection  
python train_detector.py

# Segmentation with specific action
python train_segmentation.py train --model COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml
python train_segmentation.py eval --model-path model_best.pth
python train_segmentation.py vis --n 5
```# TomatoMAP Training Suite

A modular Python project for training computer vision models on tomato phenotyping data. This project supports classification, object detection, and segmentation tasks for tomato BBCH stage analysis.

## Project Structure

```
├── main.py                 # Main entry point
├── config.py              # Configuration settings
├── utils.py               # Utility functions
├── dataset.py             # Dataset classes and data loading
├── visualize.py           # Visualization functions
├── train_classifier.py    # Classification training
├── train_detector.py      # Object detection training
├── train_segmentation.py  # Segmentation training
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Features

- **Classification (TomatoMAP-Cls)**: Train models to classify tomato BBCH stages
- **Object Detection (TomatoMAP-Det)**: Train YOLO models for tomato detection
- **Segmentation (TomatoMAP-Seg)**: Train Mask R-CNN for instance segmentation
- **Modular Design**: Easy to extend and modify
- **GPU Support**: Automatic CUDA detection and usage
- **Visualization**: Training curves and confusion matrices
- **Checkpointing**: Save and resume training

## Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd tomatomap-training
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Check your environment:
```bash
python main.py --check-env
```

Train classification model:
```bash
python main.py cls
```

Train detection model:
```bash
python main.py det
```

Train segmentation model:
```bash
python main.py seg
```

Train all models:
```bash
python main.py all
```

### Individual Training Scripts

You can also run training scripts directly:

```bash
# Classification
python train_classifier.py

# Detection  
python train_detector.py

# Segmentation (placeholder)
python train_segmentation.py
```

## Configuration

Edit `config.py` to modify training parameters:

- **CLASSIFICATION_CONFIG**: Classification model settings
- **DETECTION_CONFIG**: Object detection settings  
- **SEGMENTATION_CONFIG**: Segmentation settings (TODO)

## Data Structure

The expected data structure for classification:

```
TomatoMAP/TomatoMAP-Cls/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── val/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

For detection, provide a YOLO format dataset with the corresponding YAML file.

## Output

Training results are saved in:
- `cls/runs/` - Classification results
- `det/output/` - Detection results
- Training curves, confusion matrices, and model checkpoints

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- See `requirements.txt` for full dependencies

## Notes

- Windows users: Number of workers automatically set to 0 for compatibility
- Early stopping implemented with configurable patience
- Automatic best model saving based on validation accuracy
- Support for multiple model architectures (MobileNet, ResNet)

## TODO

- [ ] Implement segmentation training module
- [ ] Add more model architectures
- [ ] Implement inference scripts
- [ ] Add data augmentation options
- [ ] Add tensorboard logging

## Contributing

Feel free to submit issues and pull requests to improve the project.

## License

This project is licensed under the MIT License.

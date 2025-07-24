# 🍅 TomatoMAP: Tomato Multi-Angle Multi-Pose Dataset for Fine-Grained Phenotyping 
***A Novel Dataset for Tomato Fine-Grained Phenotyping*** 

<p align="center">
<a href="https://orcid.org/0009-0004-8160-809X">Yujie Zhang</a>,
<a href="">Sabine Struckmeyer</a>,
<a href="https://orcid.org/0000-0003-4753-7801">Andreas Kolb</a>,
<a href="https://orcid.org/0000-0001-9779-9610">Sven Reichardt</a>
</p>

<p align="center">
<a href="https://0yj.github.io/tomato_map/">📟[Homepage]</a>
<a href="https://arxiv.org/abs/2507.11279">📄[Paper]</a>
<a href="https://github.com/0YJ/TomatoMAP">💻[Code]</a>
<a href="https://doi.ipk-gatersleben.de/DOI/89386758-8bfd-41ca-aa9c-ee363e9d94c9/073051f0-b05e-4b43-a9cd-0435fe7cd913/2/1847940088">📁[Dataset]</a>
</p>

<p style="align:justify"><b>Abstract</b>: Observer bias and inconsistencies in traditional plant phenotyping methods limit the accuracy and reproducibility of fine-grained plant analysis. To overcome these challenges, we developed TomatoMAP, a comprehensive dataset for Solanum lycopersicum using an Internet of Things (IoT) based imaging system with standardized data acquisition protocols. Our dataset contains 64,464 RGB-images that capture 12 different plant poses from four camera elevation angles. Each image includes manually annotated bounding boxes for seven regions of interest (ROIs), including leaves, panicle, batch of flowers, batch of fruits, axillary shoot, shoot and whole plant area, along with 50 fine-grained growth stage classifications based on the BBCH scale. Additionally, we provide 3,616 high-resolution image subset with pixel-wise semantic and instance segmentation annotations. We validated our dataset using a cascading model deep learning framework combining different models. Through AI vs. Human analysis involving five domain experts, we demonstrate that the models trained on our dataset achieve accuracy comparable to the experts. Cohen's Kappa and inter-rater agreement heatmap confirm the reliability of automated fine-grained phenotyping using our approach.</p>

## 📢 Updates

* 15.07.2025: Paper available on [arXiv](https://arxiv.org/abs/2507.11279)
* 18.07.2025 Full dataset release on [e!DAL](https://doi.ipk-gatersleben.de/DOI/89386758-8bfd-41ca-aa9c-ee363e9d94c9/073051f0-b05e-4b43-a9cd-0435fe7cd913/2/1847940088)
* 23.07.2025 Code repo public


## 🌠 Coming Soon
* Code validation
* TomatoMAP builder on colab
* User friendly Web UI demo
* Claim e!DAL DOI
* Submitt to Nature Scientific Data
* Update homepage

## Getting Started
Please check [code](https://github.com/0YJ/TomatoMAP/tree/main/code) subfolder for more details. Or click: 
<details>
  <summary>Expand details</summary>

### Requirements
We suggest using [conda](https://www.anaconda.com/) for env management. 
```
conda create -n tomatomap python=3.10
conda activate tomatomap
```

```bash
# clone repo
git clone https://github.com/0YJ/TomatoMAP.git && cd TomatoMAP/code
cp det/best_hyperparameters.yaml ./

# unzip TomatoMAP dataset here

# install [PyTorch](https://pytorch.org/get-started/locally/)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install numpy pandas matplotlib tqdm pillow scikit-learn

# Task-specific requirements
# For detection:
pip install ultralytics

# For segmentation:
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install opencv-python

# For ISAT2COCO conversion:
pip install pyyaml

# Translate ISAT format to COCO
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
# Default training with MobileNetV3-Large
python main.py cls --data-dir ./TomatoMAP/TomatoMAP-Cls --epochs 100

# Options
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
# Default training with YOLO11-Large
python main.py det --data-config ./det/TomatoMAP-Det.yaml --epochs 500

# Options
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

# Analyze object size (small, big, middle)
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
</details>

## Citation

If you use TomatoMAP in your research, please cite:

```bibtex
@misc{zhang2025tomatomultianglemultiposedataset,
      title={Tomato Multi-Angle Multi-Pose Dataset for Fine-Grained Phenotyping}, 
      author={Yujie Zhang and Sabine Struckmeyer and Andreas Kolb and Sven Reichardt},
      year={2025},
      eprint={2507.11279},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.11279}, 
}


@dataset{tomatomap,
  title={TomatoMAP: Tomato Multi-Angle Multi-Pose Dataset for Fine-Grained Phenotyping},
  author={Yujie Zhang, Sabine Struckmeyer, Andreas Kolb, and Sven Reichardt},
  journal={e!DAL - Plant Genomics and Phenomics Research Data Repository (PGP)},
  year={2025}
}
```

## License

This dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Commercial use requires permission.

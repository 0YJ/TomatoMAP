# TomatoMAP Code Workspace

This README focuses on running preprocessing and training from the `code/` folder.
For dataset/paper overview, please use the root README (`../README.md`).

## Quick Start

```bash
git clone https://github.com/0YJ/TomatoMAP.git
cd TomatoMAP
conda env create --file environment.yml
conda activate TomatoMAP
cd code
```

Optional (recommended for detection fine-tuned training):

```bash
cp det/best_hyperparameters.yaml ./
```

If you need to convert ISAT annotations to COCO:

```bash
python utils/isat2coco.py
```

## Minimal Structure (code/)

```text
code/
├── main.py
├── trainers/
├── datasets/
├── models/
├── utils/
├── det/
├── cls/
└── seg/
```

## Training Commands

### Classification

```bash
python main.py cls --data-dir ./TomatoMAP/TomatoMAP-Cls --epochs 100
```

### Detection

```bash
# default detection training
python main.py det --data-config ./det/TomatoMAP-Det.yaml --epochs 500

# with custom hyperparameters
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

# class-balanced weighted sampling
python main.py det \
    --data-config ./det/TomatoMAP-Det.yaml \
    --model yolo11l.pt \
    --balanced-sampling
```

### Segmentation

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
```

## Notes

- Detection supports `--balanced-sampling` (implemented in `trainers/det_balanced_trainer.py`).
- CLI entry and authoritative arguments are defined in `main.py`.

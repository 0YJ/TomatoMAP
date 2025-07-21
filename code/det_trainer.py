#!/usr/bin/env python3
"""
TomatoMAP-Det Trainer
Detection trainer for TomatoMAP dataset using YOLO
"""

from ultralytics import YOLO
from ultralytics import RTDETR
import torch
import ultralytics

def main():
    # using proper library?
    ultralytics.checks()
    print(ultralytics.__file__)
    
    torch.use_deterministic_algorithms(False)

    print("\n" + "=" * 60)
    print("TomatoMAP-Det Trainer")
    print("\n" + "=" * 60)

    print("downloading pretrained model: ")

    model = YOLO("yolo11l.pt")

    print("model info: ")

    train_result = model.train(
        data="det/TomatoMAP-Det.yaml",
        epochs=500,
        imgsz=640,
        device=[0],
        batch=4,
        patience=10,
        project="det/output",
        cfg="det/best_hyperparameters.yaml", # fine-tuned hyperparameters, ready to use, details please contact us per email
        #profile=True,
        plots=True
    )

if __name__ == "__main__":
    main()

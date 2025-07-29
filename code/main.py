#!/usr/bin/env python3
"""
TomatoMAP Training Main Entry Point
"""

import argparse
import sys
import os
from pathlib import Path

# add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.common import print_header, check_environment
from trainers.cls_trainer import ClassificationTrainer
from trainers.det_trainer import DetectionTrainer
from trainers.seg_trainer import SegmentationTrainer


def main():
    parser = argparse.ArgumentParser(
        description='TomatoMAP Training System - Train classification, detection, and segmentation models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classification training
  python main.py cls --data-dir ./TomatoMAP/TomatoMAP-Cls --model mobilenet_v3_large --epochs 100
  
  # Detection training
  python main.py det --data-config ./det/TomatoMAP-Det.yaml --model yolo11l --epochs 500
  
  # Segmentation training
  python main.py seg train --epochs 100 --model mask_rcnn_R_50_FPN_3x.yaml
  python main.py seg eval --model-path model_best.pth
  python main.py seg vis --n 5
        """
    )
    
    # subparsers for different tasks
    subparsers = parser.add_subparsers(dest='task', help='Task to perform')
    subparsers.required = True
    
    # classification parser
    cls_parser = subparsers.add_parser('cls', help='Train classification model')
    cls_parser.add_argument('--data-dir', type=str, default='TomatoMAP/TomatoMAP-Cls',
                           help='Path to classification dataset (default: TomatoMAP/TomatoMAP-Cls)')
    cls_parser.add_argument('--model', type=str, default='mobilenet_v3_large',
                           choices=['mobilenet_v3_large', 'mobilenet_v3_small', 'mobilenet_v2', 'resnet18'],
                           help='Model architecture (default: mobilenet_v3_large)')
    cls_parser.add_argument('--num-classes', type=int, default=50,
                           help='Number of classes (default: 50)')
    cls_parser.add_argument('--batch-size', type=int, default=32,
                           help='Batch size (default: 32)')
    cls_parser.add_argument('--epochs', type=int, default=30,
                           help='Number of epochs (default: 30)')
    cls_parser.add_argument('--lr', type=float, default=1e-4,
                           help='Learning rate (default: 1e-4)')
    cls_parser.add_argument('--img-size', type=int, nargs=2, default=[640, 640],
                           help='Image size as width height (default: 640 640)')
    cls_parser.add_argument('--patience', type=int, default=3,
                           help='Early stopping patience (default: 3)')
    cls_parser.add_argument('--output-dir', type=str, default=None,
                           help='Output directory (default: outputs/cls/{model_name})')
    cls_parser.add_argument('--resume', type=str, default=None,
                           help='Resume from checkpoint')
    
    # detection parser
    det_parser = subparsers.add_parser('det', help='Train detection model')
    det_parser.add_argument('--data-config', type=str, default='det/TomatoMAP-Det.yaml',
                           help='Path to data configuration file (default: det/TomatoMAP-Det.yaml)')
    det_parser.add_argument('--model', type=str, default='yolo11l.pt',
                           help='Model name or path (default: yolo11l.pt)')
    det_parser.add_argument('--epochs', type=int, default=500,
                           help='Number of epochs (default: 500)')
    det_parser.add_argument('--img-size', type=int, default=640,
                           help='Image size (default: 640)')
    det_parser.add_argument('--batch-size', type=int, default=4,
                           help='Batch size (default: 4)')
    det_parser.add_argument('--patience', type=int, default=10,
                           help='Early stopping patience (default: 10)')
    det_parser.add_argument('--device', type=str, default='0',
                           help='Device to use (default: 0)')
    det_parser.add_argument('--output-dir', type=str, default='outputs/det',
                           help='Output directory (default: outputs/det)')
    det_parser.add_argument('--hyperparams', type=str, default=None,
                           help='Path to hyperparameters config file')
    
    # segmentation parser
    seg_parser = subparsers.add_parser('seg', help='Train/evaluate segmentation model')
    seg_subparsers = seg_parser.add_subparsers(dest='action', help='Action to perform')
    seg_subparsers.required = True
    
    # segmentation train
    seg_train = seg_subparsers.add_parser('train', help='Train segmentation model')
    seg_train.add_argument('--data-dir', type=str, default='./TomatoMAP/TomatoMAP-Seg',
                          help='Path to segmentation dataset (default: ./TomatoMAP/TomatoMAP-Seg)')
    seg_train.add_argument('--model', type=str, default='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml',
                          help='Model config file (default: mask_rcnn_R_50_FPN_1x.yaml)')
    seg_train.add_argument('--epochs', type=int, default=100,
                          help='Maximum epochs (default: 100)')
    seg_train.add_argument('--lr', type=float, default=0.0001,
                          help='Base learning rate (default: 0.0001)')
    seg_train.add_argument('--batch-size', type=int, default=4,
                          help='Batch size (default: 4)')
    seg_train.add_argument('--patience', type=int, default=5,
                          help='Early stopping patience (default: 5)')
    seg_train.add_argument('--output-dir', type=str, default='outputs/seg',
                          help='Output directory (default: outputs/seg)')
    seg_train.add_argument('--num-classes', type=int, default=10,
                          help='Number of classes (default: 10)')
    
    # segmentation eval
    seg_eval = seg_subparsers.add_parser('eval', help='Evaluate segmentation model')
    seg_eval.add_argument('--model-path', type=str, default='model_best.pth',
                         help='Model checkpoint file (default: model_best.pth)')
    seg_eval.add_argument('--output-dir', type=str, default='outputs/seg',
                         help='Output directory (default: outputs/seg)')
    seg_eval.add_argument('--data-dir', type=str, default='./TomatoMAP/TomatoMAP-Seg',
                         help='Path to segmentation dataset (default: ./TomatoMAP/TomatoMAP-Seg)')
    
    # segmentation visualize
    seg_vis = seg_subparsers.add_parser('vis', help='Visualize segmentation results')
    seg_vis.add_argument('--model-path', type=str, default='model_best.pth',
                        help='Model checkpoint file (default: model_best.pth)')
    seg_vis.add_argument('--n', type=int, default=3,
                        help='Number of images to visualize (default: 3)')
    seg_vis.add_argument('--output-dir', type=str, default='outputs/seg',
                        help='Output directory (default: outputs/seg)')
    seg_vis.add_argument('--data-dir', type=str, default='./TomatoMAP/TomatoMAP-Seg',
                        help='Path to segmentation dataset (default: ./TomatoMAP/TomatoMAP-Seg)')
    
    # segmentation info
    seg_info = seg_subparsers.add_parser('info', help='Show dataset information')
    seg_info.add_argument('--data-dir', type=str, default='./TomatoMAP/TomatoMAP-Seg',
                         help='Path to segmentation dataset (default: ./TomatoMAP/TomatoMAP-Seg)')
    
    # segmentation analyze
    seg_analyze = seg_subparsers.add_parser('analyze', help='Analyze dataset areas')
    seg_analyze.add_argument('--data-dir', type=str, default='./TomatoMAP/TomatoMAP-Seg',
                            help='Path to segmentation dataset (default: ./TomatoMAP/TomatoMAP-Seg)')
    
    args = parser.parse_args()
    
    # check environment
    check_environment()
    
    # run base on task
    if args.task == 'cls':
        print_header("Classification Training")
        trainer = ClassificationTrainer(args)
        trainer.train()
        
    elif args.task == 'det':
        print_header("Detection Training")
        trainer = DetectionTrainer(args)
        trainer.train()
        
    elif args.task == 'seg':
        print_header("Segmentation")
        trainer = SegmentationTrainer(args)
        
        if args.action == 'train':
            trainer.train()
        elif args.action == 'eval':
            trainer.evaluate()
        elif args.action == 'vis':
            trainer.visualize()
        elif args.action == 'info':
            trainer.show_info()
        elif args.action == 'analyze':
            trainer.analyze_areas()


if __name__ == "__main__":
    main()

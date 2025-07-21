#!/usr/bin/env python3
"""
Segmentation trainer for TomatoMAP dataset using Detectron2
"""

import os
import json
import random
import cv2
import numpy as np
from pathlib import Path

import torch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer

from models.seg_hooks import SegmentationTrainer as Detectron2Trainer
from utils.common import print_config, print_section, create_output_dir
from datasets.seg_dataset import analyze_dataset_areas, get_dataset_info


class SegmentationTrainer:
    """Trainer class for instance segmentation models."""
    
    def __init__(self, args):
        """Initialize trainer with arguments."""
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup paths
        self.img_dir = os.path.join(args.data_dir, 'images')
        self.ann_dir = os.path.join(args.data_dir, 'cocoOut')
        self.output_dir = Path(args.output_dir)
        
        # Check paths
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not os.path.exists(self.ann_dir):
            raise FileNotFoundError(f"Annotation directory not found: {self.ann_dir}")
        
        # Register datasets
        self._register_datasets()
        
    def _register_datasets(self):
        """Register COCO format datasets."""
        print_section("Registering datasets")
        
        # Clear existing registrations
        for split in ['train', 'val', 'test']:
            dataset_name = f'tomato_{split}'
            if dataset_name in MetadataCatalog.list():
                MetadataCatalog.remove(dataset_name)
        
        # Register datasets
        register_coco_instances(
            "tomato_train", {},
            os.path.join(self.ann_dir, "train.json"),
            self.img_dir
        )
        register_coco_instances(
            "tomato_val", {},
            os.path.join(self.ann_dir, "val.json"),
            self.img_dir
        )
        
        # Register test if exists
        test_json = os.path.join(self.ann_dir, "test.json")
        if os.path.exists(test_json):
            register_coco_instances(
                "tomato_test", {},
                test_json,
                self.img_dir
            )
            print("Registered train, val, and test datasets")
        else:
            print("Registered train and val datasets (test not found)")
    
    def _build_config(self):
        """Build Detectron2 configuration."""
        cfg = get_cfg()
        
        # Load model config
        cfg.merge_from_file(model_zoo.get_config_file(self.args.model))
        
        # Dataset configuration
        cfg.DATASETS.TRAIN = ("tomato_train",)
        cfg.DATASETS.TEST = ("tomato_val",)
        cfg.DATALOADER.NUM_WORKERS = 8
        
        # Model configuration
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.args.num_classes
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
        
        # Get pretrained weights
        weight_config = self.args.model.replace("_1x.yaml", "_3x.yaml")
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(weight_config)
        
        # Training configuration
        cfg.SOLVER.IMS_PER_BATCH = self.args.batch_size
        cfg.SOLVER.BASE_LR = self.args.lr
        
        # Estimate iterations
        estimated_iters_per_epoch = 127  # Will be updated in trainer
        cfg.SOLVER.MAX_ITER = estimated_iters_per_epoch * self.args.epochs
        cfg.SOLVER.CHECKPOINT_PERIOD = estimated_iters_per_epoch * 10
        
        # Input configuration
        cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
        cfg.INPUT.MAX_SIZE_TRAIN = 1333
        
        # Output directory
        cfg.OUTPUT_DIR = str(self.output_dir)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        
        return cfg
    
    def train(self):
        """Train segmentation model."""
        # Build configuration
        cfg = self._build_config()
        
        # Print configuration
        config = {
            'Data directory': self.args.data_dir,
            'Model config': self.args.model,
            'Number of classes': self.args.num_classes,
            'Batch size': self.args.batch_size,
            'Maximum epochs': self.args.epochs,
            'Base learning rate': self.args.lr,
            'Early stopping patience': self.args.patience,
            'Device': self.device,
            'Output directory': self.output_dir
        }
        print_config(config, "Segmentation Training Configuration")
        
        # Available models info
        print_section("Available model configurations")
        print("  - COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
        print("  - COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        print("  - COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        print("  - COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
        
        # Create trainer and start training
        print_section("Starting training")
        trainer = Detectron2Trainer(cfg, patience=self.args.patience)
        trainer.resume_or_load(resume=False)
        trainer.train()
    
    def evaluate(self):
        """Evaluate model on test set."""
        print_section("Evaluating model")
        
        # Build configuration
        cfg = self._build_config()
        
        # Set model weights
        model_path = self.output_dir / self.args.model_path
        if not model_path.exists():
            model_path = self.output_dir / "model_final.pth"
            print(f"Model {self.args.model_path} not found, using model_final.pth")
        
        if not model_path.exists():
            print(f"No trained model found in {self.output_dir}")
            return
        
        cfg.MODEL.WEIGHTS = str(model_path)
        print(f"Evaluating model: {model_path}")
        
        try:
            # Use test set if available, otherwise val
            dataset_name = "tomato_test"
            if dataset_name not in MetadataCatalog.list():
                dataset_name = "tomato_val"
                print("Test set not found, evaluating on validation set")
            
            # Create evaluator and run
            evaluator = COCOEvaluator(
                dataset_name, cfg, False,
                output_dir=str(self.output_dir / "evaluation")
            )
            val_loader = build_detection_test_loader(cfg, dataset_name)
            
            # Build model and load weights
            model = Detectron2Trainer.build_model(cfg)
            Detectron2Trainer._load_checkpoint_to_model(model, str(model_path))
            
            # Run evaluation
            results = inference_on_dataset(model, val_loader, evaluator)
            
            # Print results
            print("\nEvaluation Results:")
            print(json.dumps(results, indent=2))
            
            # Save results
            results_file = self.output_dir / f"test_results_{self.args.model_path.replace('.pth', '')}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {results_file}")
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    
    def visualize(self):
        """Visualize predictions on random samples."""
        print_section(f"Visualizing {self.args.n} random predictions")
        
        # Build configuration
        cfg = self._build_config()
        
        # Set model weights
        model_path = self.output_dir / self.args.model_path
        if not model_path.exists():
            model_path = self.output_dir / "model_final.pth"
            print(f"Model {self.args.model_path} not found, using model_final.pth")
        
        if not model_path.exists():
            print(f"No trained model found in {self.output_dir}")
            return
        
        cfg.MODEL.WEIGHTS = str(model_path)
        
        # Create predictor
        predictor = DefaultPredictor(cfg)
        
        # Get metadata
        try:
            metadata = MetadataCatalog.get("tomato_val")
        except:
            print("Warning: Could not get metadata for tomato_val")
            metadata = None
        
        # Get image list
        img_list = [f for f in os.listdir(self.img_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        
        if not img_list:
            print(f"No image files found in {self.img_dir}")
            return
        
        # Randomly sample images
        random.shuffle(img_list)
        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        shown = 0
        for img_file in img_list:
            try:
                # Load image
                img_path = os.path.join(self.img_dir, img_file)
                im = cv2.imread(img_path)
                
                if im is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                
                # Make prediction
                outputs = predictor(im)
                
                # Visualize
                v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.2)
                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                
                # Save visualization
                save_path = vis_dir / f"pred_{shown}_{img_file}"
                cv2.imwrite(str(save_path), out.get_image()[:, :, ::-1])
                print(f"Saved: {save_path}")
                
                shown += 1
                if shown >= self.args.n:
                    break
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
        
        print(f"\nVisualization complete. Results saved in: {vis_dir}")
    
    def show_info(self):
        """Show dataset information."""
        print_section("Dataset Information")
        get_dataset_info(self._build_config(), self.ann_dir)
    
    def analyze_areas(self):
        """Analyze object area distribution."""
        print_section("Analyzing Object Area Distribution")
        analyze_dataset_areas(self.ann_dir, self._build_config())

#!/usr/bin/env python3
"""
Detection trainer for TomatoMAP dataset
"""

import torch
from ultralytics import YOLO, checks
from pathlib import Path

from utils.common import create_output_dir, print_config, print_section


class DetectionTrainer:
    
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        
        # disable deterministic algorithms for YOLO
        torch.use_deterministic_algorithms(False)
        
    def train(self):
        # check environment
        print_section("YOLO Environment Check")
        checks()
        
        # print configuration
        config = {
            'Data config': self.args.data_config,
            'Model': self.args.model,
            'Epochs': self.args.epochs,
            'Image size': self.args.img_size,
            'Batch size': self.args.batch_size,
            'Device': self.args.device,
            'Patience': self.args.patience,
            'Output directory': self.args.output_dir,
            'Hyperparameters': self.args.hyperparams or 'Default'
        }
        print_config(config, "Detection Training Configuration")
        
        # load model
        print_section("Loading model")
        print(f"Loading model: {self.args.model}")
        model = YOLO(self.args.model)
        
        # prepare training arguments
        train_args = {
            'data': self.args.data_config,
            'epochs': self.args.epochs,
            'imgsz': self.args.img_size,
            'batch': self.args.batch_size,
            'patience': self.args.patience,
            'project': str(self.output_dir),
            'name': 'train',
            'exist_ok': True,
            'plots': True,
            'save': True,
            'save_period': -1,  # save best and last only
        }
        
        # add device configuration
        if self.args.device.lower() == 'cpu':
            train_args['device'] = 'cpu'
        else:
            # handle multiple GPUs
            if ',' in self.args.device:
                devices = [int(d) for d in self.args.device.split(',')]
                train_args['device'] = devices
            else:
                train_args['device'] = int(self.args.device)
        
        # add hyperparameters (we offer fine-tuned hyperparameters)
        if self.args.hyperparams:
            train_args['cfg'] = self.args.hyperparams
            print(f"Using custom hyperparameters: {self.args.hyperparams}")
        
        # start training
        print_section("Starting training")
        results = model.train(**train_args)
        
        # print results
        print_section("Training completed")
        print(f"Results saved in: {self.output_dir / 'train'}")
        print("\nTraining metrics:")
        if hasattr(results, 'results_dict'):
            for key, value in results.results_dict.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
        
        # evaluate on test set if possible
        print_section("Evaluating on test set")
        try:
            test_results = model.val(
                data=self.args.data_config,
                split='test',
                save_json=True,
                save_hybrid=True,
                project=str(self.output_dir),
                name='test',
                exist_ok=True
            )
            
            print("\nTest results:")
            if hasattr(test_results, 'results_dict'):
                for key, value in test_results.results_dict.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
        except Exception as e:
            print(f"Test evaluation failed: {e}")
            print("This is normal if test set is not available")
        
        print(f"\nModel weights saved at:")
        print(f"  Best: {self.output_dir / 'train' / 'weights' / 'best.pt'}")
        print(f"  Last: {self.output_dir / 'train' / 'weights' / 'last.pt'}")

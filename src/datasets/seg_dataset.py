#!/usr/bin/env python3
"""
Dataset utilities for TomatoMAP-Seg
"""

import json
import numpy as np
from detectron2.data import build_detection_train_loader

from utils.common import print_section


def get_dataset_info(cfg, ann_dir=None):
    try:
        train_loader = build_detection_train_loader(cfg)
        train_size = len(train_loader)
        
        print(f"Training dataset size: {train_size} images")
        print(f"Images per batch: {cfg.SOLVER.IMS_PER_BATCH}")
        print(f"Iterations per epoch: {train_size // cfg.SOLVER.IMS_PER_BATCH}")
        print(f"Max iterations: {cfg.SOLVER.MAX_ITER}")
        print(f"Estimated epochs: {cfg.SOLVER.MAX_ITER // (train_size // cfg.SOLVER.IMS_PER_BATCH)}")
        
        return train_size
    except Exception as e:
        print(f"Could not determine dataset size: {e}")
        return None


def analyze_dataset_areas(ann_dir, cfg):
    print("Analyzing object area distribution...")
    print("=" * 60)
    
    # get input size parameters
    min_size = min(cfg.INPUT.MIN_SIZE_TRAIN)
    max_size = cfg.INPUT.MAX_SIZE_TRAIN
    
    for split in ['train', 'val', 'test']:
        ann_file = f"{ann_dir}/{split}.json"
        try:
            with open(ann_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"\n{split.upper()} set: Not found")
            continue
        
        # create image ID to info mapping
        image_info = {img['id']: img for img in data['images']}
        
        areas_original = []
        areas_scaled = []
        
        # process annotations
        for ann in data['annotations']:
            # Original area
            if 'area' in ann:
                area = ann['area']
            else:
                bbox = ann.get('bbox', [0, 0, 0, 0])
                area = bbox[2] * bbox[3]
            areas_original.append(area)
            
            # estimate scaled area
            img_id = ann['image_id']
            if img_id in image_info:
                img = image_info[img_id]
                orig_w, orig_h = img['width'], img['height']
                
                # simulate Detectron2 scaling logic
                size = max(orig_w, orig_h)
                if size > max_size:
                    scale = max_size / size
                else:
                    scale = min_size / min(orig_w, orig_h)
                    if scale * size > max_size:
                        scale = max_size / size
                
                scaled_area = area * (scale ** 2)
                areas_scaled.append(scaled_area)
        
        areas_original = np.array(areas_original)
        areas_scaled = np.array(areas_scaled) if areas_scaled else areas_original
        
        print(f"\n{split.upper()} Set Analysis:")
        print("-" * 40)
        
        # image statistics
        if len(data['images']) > 0:
            avg_width = np.mean([img['width'] for img in data['images']])
            avg_height = np.mean([img['height'] for img in data['images']])
            print(f"Average image size: {avg_width:.0f} x {avg_height:.0f}")
        
        print(f"Total objects: {len(areas_original)}")
        
        # original distribution
        print(f"\nOriginal image object areas:")
        small_orig = np.sum(areas_original < 32**2)
        medium_orig = np.sum((areas_original >= 32**2) & (areas_original < 96**2))
        large_orig = np.sum(areas_original >= 96**2)
        
        print(f"  Small (<32²): {small_orig} ({small_orig/len(areas_original)*100:.1f}%)")
        print(f"  Medium (32²-96²): {medium_orig} ({medium_orig/len(areas_original)*100:.1f}%)")
        print(f"  Large (>96²): {large_orig} ({large_orig/len(areas_original)*100:.1f}%)")
        print(f"  Min area: {np.min(areas_original):.0f} pixels²")
        print(f"  Max area: {np.max(areas_original):.0f} pixels²")
        print(f"  Mean area: {np.mean(areas_original):.0f} pixels²")
        
        # scaled distribution
        print(f"\nAfter scaling to {min_size}-{max_size}:")
        small_scaled = np.sum(areas_scaled < 32**2)
        medium_scaled = np.sum((areas_scaled >= 32**2) & (areas_scaled < 96**2))
        large_scaled = np.sum(areas_scaled >= 96**2)
        
        print(f"  Small (<32²): {small_scaled} ({small_scaled/len(areas_scaled)*100:.1f}%)")
        print(f"  Medium (32²-96²): {medium_scaled} ({medium_scaled/len(areas_scaled)*100:.1f}%)")
        print(f"  Large (>96²): {large_scaled} ({large_scaled/len(areas_scaled)*100:.1f}%)")
        
        # warnings
        if small_scaled == 0:
            print(f"\n No small objects after scaling - APs metric will be -1")
        if medium_scaled == 0:
            print(f" No medium objects after scaling - APm metric will be -1")
        if large_scaled == 0:
            print(f" No large objects after scaling - APl metric will be -1")

#!/usr/bin/env python3
"""
ISAT to COCO format converter for segmentation annotations
Converts ISAT annotation format to COCO format for Detectron2 training
"""

import os
import json
import yaml
import random
import argparse
from tqdm import tqdm
from collections import defaultdict

def flatten_segmentation(points):
    """convert [[x,y],[x,y]] → [x1,y1,x2,y2,...]"""
    return [coord for pair in points for coord in pair]

def load_categories(yaml_path):
    """load categories from ISAT yaml file"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    categories = []
    cat_map = {}
    cat_id = 1
    for item in data['label']:
        name = item['name']
        if name == '__background__':
            continue
        categories.append({
            "id": cat_id,
            "name": name,
            "supercategory": "none"
        })
        cat_map[name] = cat_id
        cat_id += 1
    return categories, cat_map

def convert_isat_folder_to_coco(task_dir, label_dir, yaml_path, output_dir, 
                               train_ratio=0.7, val_ratio=0.2):
    """convert ISAT annotations to COCO format"""
    os.makedirs(output_dir, exist_ok=True)

    categories, category_map = load_categories(yaml_path)

    # match images with annotations
    images = [f for f in os.listdir(task_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    json_map = {os.path.splitext(f)[0]: f for f in os.listdir(label_dir) if f.endswith(".json")}

    dataset = []
    for img_name in tqdm(images, desc="matching images and annotations"):
        base = os.path.splitext(img_name)[0]
        if base in json_map:
            dataset.append({
                "img_file": img_name,
                "json_file": json_map[base]
            })

    # shuffle and split dataset
    random.shuffle(dataset)
    total = len(dataset)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    splits = {
        "train": dataset[:train_end],
        "val": dataset[train_end:val_end],
        "test": dataset[val_end:]
    }

    # convert each split
    for split_name, split_data in splits.items():
        coco = {
            "images": [],
            "annotations": [],
            "categories": categories
        }
        ann_id = 1
        img_id = 1

        for item in tqdm(split_data, desc=f"converting {split_name}"):
            img_path = os.path.join(task_dir, item["img_file"])
            json_path = os.path.join(label_dir, item["json_file"])

            with open(json_path, 'r') as f:
                isat = json.load(f)

            info = isat['info']
            coco["images"].append({
                "file_name": item["img_file"],
                "id": img_id,
                "width": info["width"],
                "height": info["height"]
            })

            # process annotations
            for obj in isat['objects']:
                cat = obj['category']
                if cat not in category_map:
                    continue
                
                # fix segmentation point format
                seg_flat = flatten_segmentation(obj["segmentation"])
                if len(seg_flat) < 6:
                    continue  # skip invalid polygons (less than 3 points)

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_map[cat],
                    "segmentation": [seg_flat],
                    "bbox": obj["bbox"],
                    "area": obj["area"],
                    "iscrowd": obj.get("iscrowd", 0),
                    "group_id": obj.get("group", None)
                })
                ann_id += 1
            img_id += 1

        # save COCO format file
        output_file = os.path.join(output_dir, f"{split_name}.json")
        with open(output_file, "w

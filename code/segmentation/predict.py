import os
import cv2
import argparse
import torch
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

NUM_CLASSES = 10
OUTPUT_DIR = "./output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "model_final.pth")
ISAT_YAML_PATH = "isat.yaml"
IMG_EXT = ['.jpg', '.jpeg', '.png', '.bmp']

def load_class_labels(yaml_path):
    import yaml
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return [item['name'] for item in data['label'] if item['name'] != '__background__']

def build_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.INPUT.MIN_SIZE_TEST = 1100
    cfg.INPUT.MAX_SIZE_TEST = 1600
    cfg.MODEL.WEIGHTS = MODEL_PATH
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg

def predict_image(predictor, metadata, image_path, save_dir):
    image = cv2.imread(image_path)
    outputs = predictor(image)

    v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.2)
    v._default_font_size = 12
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    save_path = os.path.join(save_dir, f"pred_{os.path.basename(image_path)}")
    cv2.imwrite(save_path, out.get_image()[:, :, ::-1])
    print(f"[Saved] {save_path}")

def main(input_path):
    cfg = build_cfg()
    labels = load_class_labels(ISAT_YAML_PATH)
    MetadataCatalog.get("inference").thing_classes = labels
    metadata = MetadataCatalog.get("inference")
    predictor = DefaultPredictor(cfg)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.isdir(input_path):
        for fname in os.listdir(input_path):
            if any(fname.lower().endswith(ext) for ext in IMG_EXT):
                predict_image(predictor, metadata, os.path.join(input_path, fname), OUTPUT_DIR)
    elif os.path.isfile(input_path):
        predict_image(predictor, metadata, input_path, OUTPUT_DIR)
    else:
        print(f"[Error] Invalid input: {input_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Image file or directory path")
    args = parser.parse_args()
    main(args.input)

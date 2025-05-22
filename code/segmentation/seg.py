import os
import cv2
import json
import yaml
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.ops import box_iou
import torch
from sklearn.metrics import confusion_matrix
from pycocotools import mask as mask_utils
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets.coco import load_coco_json

DATASET_ROOT = "./"
IMG_DIR = os.path.join(DATASET_ROOT, "mixed")
ANN_DIR = os.path.join(DATASET_ROOT, "cocoOut")
OUTPUT_DIR = "./output"
NUM_CLASSES = 10
ISAT_YAML_PATH = "isat.yaml"

def load_class_labels(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    labels = data['label']
    return [item['name'] for item in labels if item['name'] != '__background__']

def register_all_datasets():
    labels = load_class_labels(ISAT_YAML_PATH)

    register_coco_instances("tomato_train", {}, os.path.join(ANN_DIR, "train.json"), IMG_DIR)
    register_coco_instances("tomato_val", {}, os.path.join(ANN_DIR, "val.json"), IMG_DIR)
    register_coco_instances("tomato_test", {}, os.path.join(ANN_DIR, "test.json"), IMG_DIR)

    for dset in ["tomato_train", "tomato_val", "tomato_test"]:
        MetadataCatalog.get(dset).thing_classes = labels

def build_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("tomato_train",)
    cfg.DATASETS.TEST = ("tomato_test",)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.OUTPUT_DIR = OUTPUT_DIR
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

def visualize_all_predictions(dataset_name="tomato_test"):
    cfg = build_cfg()
    cfg.MODEL.WEIGHTS = os.path.join(OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(dataset_name)
    dataset_dicts = load_coco_json(os.path.join(ANN_DIR, f"{dataset_name.split('_')[-1]}.json"), IMG_DIR)

    for d in dataset_dicts:
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.2)
        v._default_font_size = 150
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        save_path = os.path.join(OUTPUT_DIR, f"prediction_{os.path.basename(d['file_name'])}")
        cv2.imwrite(save_path, out.get_image()[:, :, ::-1])
        print(f"Saved: {save_path}")

def visualize_segmentation(dataset_name="tomato_test"):
    cfg = build_cfg()
    cfg.MODEL.WEIGHTS = os.path.join(OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(dataset_name)
    labels = metadata.thing_classes
    rng = np.random.RandomState(42)
    colors = rng.randint(0, 255, size=(len(labels), 3), dtype=np.uint8)

    dataset_dicts = load_coco_json(os.path.join(ANN_DIR, f"{dataset_name.split('_')[-1]}.json"), IMG_DIR)

    for d in dataset_dicts:
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        instances = outputs["instances"].to("cpu")
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.2)
        #v._default_font_size = 150
        for i in range(len(instances)):
            box = instances.pred_boxes[i].tensor.numpy()[0].astype(int)
            class_id = int(instances.pred_classes[i])
            score = float(instances.scores[i]) if instances.has("scores") else None
            mask = instances.pred_masks[i].numpy()
            color = [c / 255.0 for c in colors[class_id]]
            label = f"{labels[class_id]} {score:.2f}" if score is not None else labels[class_id]
            v.draw_box(box.tolist(), edge_color=color)
            v.draw_binary_mask(mask, color=color, alpha=0.3)
            v.draw_text(label, (box[0], max(box[1] - 5, 0)), color=color, font_size=150)
        final = v.output.get_image()[:, :, ::-1]
        save_path = os.path.join(cfg.OUTPUT_DIR, f"seg_{os.path.basename(d['file_name'])}")
        cv2.imwrite(save_path, final)
        print(f"Saved: {save_path}")

def generate_confusion_matrix(dataset_name="tomato_val"):
    cfg = build_cfg()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.WEIGHTS = os.path.join(OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(cfg)

    dataset_dicts = load_coco_json(
        os.path.join(ANN_DIR, f"{dataset_name.split('_')[-1]}.json"),
        IMG_DIR,
        dataset_name=dataset_name
    )

    metadata = MetadataCatalog.get(dataset_name)
    class_names = metadata.thing_classes
    num_classes = len(class_names)

    cmatrix_total = np.zeros((num_classes, num_classes), dtype=np.int64)

    for data in dataset_dicts:
        height, width = data["height"], data["width"]
        image = cv2.imread(data["file_name"])
        if image is None:
            continue

        gt_mask = np.zeros((height, width), dtype=np.uint8)
        for ann in data.get("annotations", []):
            category_id = ann["category_id"]
            segmentation = ann["segmentation"]
            if isinstance(segmentation, list):
                rle = mask_utils.frPyObjects(segmentation, height, width)
                rle = mask_utils.merge(rle)
            elif isinstance(segmentation, dict) and "counts" in segmentation:
                rle = segmentation
            else:
                continue
            m = mask_utils.decode(rle)
            gt_mask[m == 1] = category_id

        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")

        pred_mask = np.zeros((height, width), dtype=np.uint8)
        for i in range(len(instances)):
            class_id = int(instances.pred_classes[i])
            mask = instances.pred_masks[i].numpy()
            pred_mask[mask == 1] = class_id

        cm_local = confusion_matrix(
            gt_mask.flatten(),
            pred_mask.flatten(),
            labels=list(range(num_classes))
        )
        cmatrix_total += cm_local

    cmatrix_norm = np.nan_to_num(cmatrix_total.astype('float') / cmatrix_total.sum(axis=1, keepdims=True))
    df = pd.DataFrame(cmatrix_norm, index=class_names, columns=class_names)
    df.to_excel(os.path.join(OUTPUT_DIR, f"confmat_instance_{dataset_name}.xlsx"))

    fig, ax = plt.subplots(figsize=(8, 8))
    masked = np.ma.masked_where(cmatrix_norm == 0, cmatrix_norm)
    im = ax.imshow(masked, cmap="jet", vmin=0.0, vmax=1.0)
    for i in range(num_classes):
        for j in range(num_classes):
            val = cmatrix_norm[i, j]
            if val > 0:
                color = 'white' if val < 0.5 or val > 0.9 else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=color)
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Truth")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"confmat_{dataset_name}.pdf"))
    print(f"[INFO] Confusion matrix saved to {OUTPUT_DIR}")

def train():
    cfg = build_cfg()
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

def evaluate(dataset_name="tomato_test"):
    cfg = build_cfg()
    cfg.MODEL.WEIGHTS = os.path.join(OUTPUT_DIR, "model_final.pth")
    evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, dataset_name)
    results = inference_on_dataset(DefaultTrainer.build_model(cfg), val_loader, evaluator)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "eval", "eval_val", "vis", "confmat", "seg"], help="choose: train / eval / eval_val / vis / confmat / seg")
    args = parser.parse_args()

    register_all_datasets()

    if args.action == "train":
        train()
    elif args.action == "eval":
        evaluate("tomato_test")
    elif args.action == "eval_val":
        evaluate("tomato_val")
    elif args.action == "vis":
        visualize_all_predictions("tomato_test")
    elif args.action == "seg":
        visualize_segmentation("tomato_val")
    elif args.action == "confmat":
        generate_confusion_matrix("tomato_val")

#!/usr/bin/env python3
"""
Balanced detection trainer for TomatoMAP dataset
"""

from pathlib import Path

import numpy as np
import torch
from ultralytics import RTDETR, YOLO, checks
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.yolo.detect.train import DetectionTrainer as UltralyticsDetectionTrainer

from utils.common import print_config, print_section


class YOLOWeightedDataset(YOLODataset):

    def __init__(self, *args, mode="train", **kwargs):
        super().__init__(*args, **kwargs)
        self.train_mode = "train" in self.prefix

        self.count_instances()
        class_weights = np.sum(self.counts) / self.counts

        self.agg_func = np.mean
        self.class_weights = np.array(class_weights)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()

        self.print_statistics()

    def count_instances(self):
        self.counts = [0 for _ in range(len(self.data["names"]))]
        for label in self.labels:
            classes = label["cls"].reshape(-1).astype(int)
            for class_id in classes:
                self.counts[class_id] += 1
        self.counts = np.array(self.counts)
        self.counts = np.where(self.counts == 0, 1, self.counts)

    def calculate_weights(self):
        weights = []
        for label in self.labels:
            classes = label["cls"].reshape(-1).astype(int)
            if classes.size == 0:
                weights.append(1)
                continue
            weight = self.agg_func(self.class_weights[classes])
            weights.append(weight)
        return weights

    def calculate_probabilities(self):
        total_weight = sum(self.weights)
        return [weight / total_weight for weight in self.weights]

    def print_statistics(self):
        print("\n" + "=" * 80)
        print(f"Weighted Sampling Statistics ({self.prefix.strip()})")
        print("=" * 80)
        print(f"{'Class':<5} {'Name':<20} {'Samples':>10} {'Weight':>10} {'Ratio':>10}")
        print("-" * 80)
        class_names = self.data["names"]
        for index, (count, weight) in enumerate(zip(self.counts, self.class_weights)):
            name = class_names[index] if index < len(class_names) else f"class_{index}"
            ratio = count / np.sum(self.counts) * 100
            print(f"{index:<5} {name:<20} {count:>10} {weight:>10.2f} {ratio:>9.1f}%")
        print("=" * 80)
        print(f"Total: {np.sum(self.counts)} instances, {len(self.labels)} images")
        print(f"Weighted sampling: {'ON' if self.train_mode else 'OFF'}")
        print("=" * 80 + "\n")

    def __getitem__(self, index):
        if not self.train_mode:
            return self.transforms(self.get_image_and_label(index))

        sampled_index = np.random.choice(len(self.labels), p=self.probabilities)
        return self.transforms(self.get_image_and_label(sampled_index))


class WeightedDetectionTrainer(UltralyticsDetectionTrainer):

    def build_dataset(self, img_path, mode="train", batch=None):
        stride = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)
        return YOLOWeightedDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect if mode == "train" else True,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(stride),
            pad=0.0 if mode == "train" else 0.5,
            prefix=f"{mode}: ",
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )


class WeightedRTDETRTrainer(RTDETRTrainer):

    def build_dataset(self, img_path, mode="train", batch=None):
        stride = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)
        return YOLOWeightedDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False if mode == "train" else True,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(stride),
            pad=0.0 if mode == "train" else 0.5,
            prefix=f"{mode}: ",
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )


class BalancedDetectionTrainer:

    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        torch.use_deterministic_algorithms(False)

    @staticmethod
    def _is_rtdetr_model(model_name: str) -> bool:
        return "rtdetr" in model_name.lower()

    @staticmethod
    def _parse_device(device: str):
        if device.lower() == "cpu":
            return "cpu"
        if "," in device:
            return [int(item.strip()) for item in device.split(",") if item.strip()]
        return int(device)

    def _resolve_device(self):
        if not torch.cuda.is_available() and self.args.device.lower() != "cpu":
            print("CUDA not available, fallback to CPU")
            return "cpu"
        return self.args.device

    def train(self):
        resolved_device = self._resolve_device()

        print_section("YOLO Environment Check")
        checks()

        config = {
            "Data config": self.args.data_config,
            "Model": self.args.model,
            "Epochs": self.args.epochs,
            "Image size": self.args.img_size,
            "Batch size": self.args.batch_size,
            "Device": resolved_device,
            "Patience": self.args.patience,
            "Output directory": self.args.output_dir,
            "Hyperparameters": self.args.hyperparams or "Default",
            "Weighted sampling": "Enabled",
        }
        print_config(config, "Balanced Detection Training Configuration")

        print_section("Loading model")
        print(f"Loading model: {self.args.model}")
        if self._is_rtdetr_model(self.args.model):
            model = RTDETR(self.args.model)
            trainer_cls = WeightedRTDETRTrainer
        else:
            model = YOLO(self.args.model)
            trainer_cls = WeightedDetectionTrainer

        train_args = {
            "data": self.args.data_config,
            "epochs": self.args.epochs,
            "imgsz": self.args.img_size,
            "batch": self.args.batch_size,
            "patience": self.args.patience,
            "project": str(self.output_dir),
            "name": "train",
            "exist_ok": True,
            "plots": True,
            "save": True,
            "save_period": -1,
            "device": self._parse_device(resolved_device),
            "trainer": trainer_cls,
        }

        if self.args.hyperparams:
            train_args["cfg"] = self.args.hyperparams
            print(f"Using custom hyperparameters: {self.args.hyperparams}")

        print_section("Starting balanced training")
        results = model.train(**train_args)

        print_section("Training completed")
        print(f"Results saved in: {self.output_dir / 'train'}")
        print("\nTraining metrics:")
        if hasattr(results, "results_dict"):
            for key, value in results.results_dict.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")

        print_section("Evaluating on test set")
        try:
            test_results = model.val(
                data=self.args.data_config,
                split="test",
                save_json=True,
                save_hybrid=True,
                project=str(self.output_dir),
                name="test",
                exist_ok=True,
            )

            print("\nTest results:")
            if hasattr(test_results, "results_dict"):
                for key, value in test_results.results_dict.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
        except Exception as exc:
            print(f"Test evaluation failed: {exc}")
            print("This is normal if test set is not available")

        print("\nModel weights saved at:")
        print(f"  Best: {self.output_dir / 'train' / 'weights' / 'best.pt'}")
        print(f"  Last: {self.output_dir / 'train' / 'weights' / 'last.pt'}")

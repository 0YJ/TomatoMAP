import os
import cv2
import random
import torch
import json
import numpy as np
from collections import OrderedDict

from detectron2.engine import DefaultTrainer, HookBase
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.events import get_event_storage

DATASET_ROOT = "./"
IMG_DIR = os.path.join(DATASET_ROOT, "TomatoMAP/TomatoMAP-Seg/images")
ANN_DIR = os.path.join(DATASET_ROOT, "TomatoMAP/TomatoMAP-Seg/cocoOut")
DEFAULT_OUTPUT_DIR = "./output"  # 默认输出目录
NUM_CLASSES = 10
MAX_EPOCHS = 100  # 设置最大epoch数
PATIENCE = 5     # 早停patience

# 检查数据集路径是否存在
if not os.path.exists(IMG_DIR):
    print(f"Warning: Image directory {IMG_DIR} does not exist!")
if not os.path.exists(ANN_DIR):
    print(f"Warning: Annotation directory {ANN_DIR} does not exist!")

# 注册数据集
register_coco_instances("tomato_train", {}, os.path.join(ANN_DIR, "train.json"), IMG_DIR)
register_coco_instances("tomato_val", {}, os.path.join(ANN_DIR, "val.json"), IMG_DIR)
register_coco_instances("tomato_test", {}, os.path.join(ANN_DIR, "test.json"), IMG_DIR)

class BestModelHook(HookBase):
    """Hook to save the best model based on validation segmentation mAP"""
    
    def __init__(self, cfg, eval_period, patience=10):
        self.cfg = cfg.clone()
        self.eval_period = eval_period
        self.patience = patience
        self.best_score = 0  # 使用0而不是-1
        self.best_metric_name = None
        self.best_epoch = -1
        self.epochs_without_improvement = 0
        self.should_stop = False
        self.history = []  # 记录历史
        
    def get_valid_score(self, segm_results):
        """获取有效的评估分数，忽略-1值"""
        # 按优先级尝试不同的指标
        priority_metrics = ["AP", "AP50", "AP75", "APm", "APl"]
        
        for metric in priority_metrics:
            value = segm_results.get(metric, -1)
            if value != -1:  # 只返回有效值
                return metric, value
        
        return None, None
    
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final_iter = next_iter == self.trainer.max_iter
        
        if (next_iter % self.eval_period == 0 and not is_final_iter):
            current_epoch = (next_iter // self.eval_period)
            
            # 执行评估
            results = self._do_eval()
            if results is None:
                print(f"Epoch {current_epoch}: Evaluation failed")
                return
            
            segm_results = results.get("segm", {})
            bbox_results = results.get("bbox", {})
            
            # 获取有效分数
            metric_name, current_score = self.get_valid_score(segm_results)
            
            # 打印所有结果
            print(f"\n{'='*60}")
            print(f"Epoch {current_epoch} Evaluation Results:")
            print(f"{'='*60}")
            
            # 打印bbox指标
            print("\nBounding Box Metrics:")
            for key in ["AP", "AP50", "AP75", "APs", "APm", "APl"]:
                value = bbox_results.get(key, -1)
                if value != -1:
                    print(f"  {key}: {value:.4f} ✓")
                else:
                    print(f"  {key}: N/A")
            
            # 打印segmentation指标
            print("\nSegmentation Metrics:")
            for key in ["AP", "AP50", "AP75", "APs", "APm", "APl"]:
                value = segm_results.get(key, -1)
                if value != -1:
                    print(f"  {key}: {value:.4f} ✓")
                else:
                    print(f"  {key}: N/A (no objects in this category)")
            
            # 如果没有有效指标
            if metric_name is None:
                print("\n⚠️ Warning: No valid metrics available for evaluation!")
                print("This might happen if your dataset has no objects after augmentation.")
                # 尝试使用bbox指标作为备选
                metric_name, current_score = self.get_valid_score(bbox_results)
                if metric_name is not None:
                    print(f"Using bbox metric instead: {metric_name} = {current_score:.4f}")
                else:
                    return
            
            print(f"\n Primary metric: {metric_name} = {current_score:.4f}")
            
            # 记录历史
            self.history.append({
                'epoch': current_epoch,
                'metric': metric_name,
                'score': current_score,
                'all_metrics': {**segm_results, **{'bbox_' + k: v for k, v in bbox_results.items()}}
            })
            
            # 检查是否改善
            if current_score > self.best_score:
                improvement = current_score - self.best_score
                self.best_score = current_score
                self.best_metric_name = metric_name
                self.best_epoch = current_epoch
                self.epochs_without_improvement = 0
                
                # 保存模型
                self.trainer.checkpointer.save("model_best")
                print(f"\n🎉 New best model saved!")
                print(f"   Score: {current_score:.4f} (↑{improvement:.4f})")
                
                # 保存评估结果
                best_results_file = os.path.join(self.cfg.OUTPUT_DIR, "best_results.json")
                with open(best_results_file, 'w') as f:
                    json.dump({
                        'epoch': current_epoch,
                        'metric': metric_name,
                        'score': current_score,
                        'segm_results': segm_results,
                        'bbox_results': bbox_results
                    }, f, indent=2)
            else:
                self.epochs_without_improvement += 1
                gap = self.best_score - current_score
                print(f"\nCurrent: {current_score:.4f} | Best: {self.best_score:.4f} (gap: {gap:.4f})")
                print(f"No improvement for {self.epochs_without_improvement}/{self.patience} epochs")
            
            # 早停检查
            if self.epochs_without_improvement >= self.patience:
                print(f"\n{'='*60}")
                print(f"EARLY STOPPING TRIGGERED")
                print(f"   Best {self.best_metric_name}: {self.best_score:.4f} at epoch {self.best_epoch}")
                print(f"   Total epochs trained: {current_epoch}")
                print(f"{'='*60}")
                self.should_stop = True
                # 立即停止训练
                self.trainer.storage._iter = self.trainer.max_iter
    
    def _do_eval(self):
        """执行验证评估"""
        try:
            evaluator = COCOEvaluator("tomato_val", self.cfg, False, 
                                    output_dir=os.path.join(self.cfg.OUTPUT_DIR, "inference"))
            val_loader = build_detection_test_loader(self.cfg, "tomato_val")
            results = inference_on_dataset(self.trainer.model, val_loader, evaluator)
            return results
        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

class MyTrainer(DefaultTrainer):
    def __init__(self, cfg, patience=None):
        #super().__init__(cfg)
        self.patience = patience if patience is not None else PATIENCE
        super().__init__(cfg)
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(
            dataset_name=dataset_name,
            distributed=False,
            output_dir=output_folder,
            use_fast_impl=True,
            tasks=("bbox", "segm"),
        )
    
    def build_hooks(self):
        hooks = super().build_hooks()
        
        # 计算每个epoch的迭代次数
        try:
            train_loader = build_detection_train_loader(self.cfg)
            iters_per_epoch = len(train_loader) // self.cfg.SOLVER.IMS_PER_BATCH
            print(f"Iterations per epoch: {iters_per_epoch}")
        except:
            # 如果无法获取确切数量，使用估算值
            iters_per_epoch = 127
            print(f"Using estimated iterations per epoch: {iters_per_epoch}")
        
        eval_period = iters_per_epoch * 10  # 每10个epoch评估一次,5个patient
        
        # 添加最佳模型保存和早停hook
        best_model_hook = BestModelHook(self.cfg, eval_period, self.patience)
        hooks.append(best_model_hook)
        
        # 存储hook引用以便检查早停条件
        self.best_model_hook = best_model_hook
        
        return hooks
    
    def run_step(self):
        """重写 run_step 以支持早停"""
        super().run_step()
        
        # 检查是否需要早停
        if hasattr(self, 'best_model_hook') and self.best_model_hook.should_stop:
            print("Early stopping condition met. Stopping training...")
            self.storage._iter = self.max_iter
    
    def train(self):
        """重写train方法以支持早停"""
        super().train()
        
        # 训练结束后打印最佳结果
        if hasattr(self, 'best_model_hook'):
            print(f"\n{'='*60}")
            print(f"Training Summary:")
            print(f"{'='*60}")
            if self.best_model_hook.best_score > 0:
                print(f"Best {self.best_model_hook.best_metric_name}: {self.best_model_hook.best_score:.4f}")
                print(f"Best epoch: {self.best_model_hook.best_epoch}")
                print(f"Best model saved as: model_best.pth")
            else:
                print("No valid metrics were found during training.")
            
            # 保存训练历史
            history_file = os.path.join(self.cfg.OUTPUT_DIR, "training_history.json")
            with open(history_file, 'w') as f:
                json.dump(self.best_model_hook.history, f, indent=2)
            print(f"Training history saved to: {history_file}")

def build_cfg(model_config=None, base_lr=None, max_epochs=None, output_dir=None):
    cfg = get_cfg()
    
    # 使用指定的模型配置，如果没有则使用默认
    if model_config is None:
        model_config = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
    
    cfg.merge_from_file(model_zoo.get_config_file(model_config))

    cfg.DATASETS.TRAIN = ("tomato_train",)
    cfg.DATASETS.TEST = ("tomato_val",)
    cfg.DATALOADER.NUM_WORKERS = 8

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = base_lr if base_lr is not None else 0.0001
    
    # 对于5K图像，可以考虑使用更大的训练尺寸
    # 默认设置
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    
    # 使用epoch而不是iter - 估算总迭代次数
    # 根据实际训练集大小调整
    estimated_iters_per_epoch = 127
    epochs = max_epochs if max_epochs is not None else MAX_EPOCHS
    cfg.SOLVER.MAX_ITER = estimated_iters_per_epoch * epochs
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1

    # 根据模型配置文件获取对应的预训练权重
    # 将配置文件名中的yaml替换掉以匹配权重URL
    weight_config = model_config.replace("_1x.yaml", "_3x.yaml").replace("_3x.yaml", "_3x.yaml")
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(weight_config)
    
    cfg.OUTPUT_DIR = output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # 设置checkpoint保存周期（每5个epoch保存一次）
    cfg.SOLVER.CHECKPOINT_PERIOD = estimated_iters_per_epoch * 10

    return cfg

def train(model_config=None, base_lr=None, max_epochs=None, patience=None, output_dir=None):
    """训练函数"""
    cfg = build_cfg(model_config=model_config, base_lr=base_lr, max_epochs=max_epochs, output_dir=output_dir)
    trainer = MyTrainer(cfg, patience=patience)
    trainer.resume_or_load(resume=False)
    
    epochs = max_epochs if max_epochs is not None else MAX_EPOCHS
    pat = patience if patience is not None else PATIENCE
    
    print(f"\n{'='*60}")
    print(f"Starting training with the following settings:")
    print(f"  Model: {model_config or 'mask_rcnn_R_50_FPN_1x.yaml'}")
    print(f"  Maximum epochs: {epochs}")
    print(f"  Early stopping patience: {pat} epochs")
    print(f"  Output directory: {cfg.OUTPUT_DIR}")
    print(f"  Images per batch: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"  Base learning rate: {cfg.SOLVER.BASE_LR}")
    print(f"  Input size range: {cfg.INPUT.MIN_SIZE_TRAIN[0]}-{cfg.INPUT.MAX_SIZE_TRAIN}")
    print(f"{'='*60}\n")
    
    trainer.train()

def evaluate(model_path="model_best.pth", output_dir=None):
    """在测试集上评估模型"""
    cfg = build_cfg(output_dir=output_dir)  # 使用默认配置
    out_dir = output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR
    model_file = os.path.join(out_dir, model_path)
    
    if not os.path.exists(model_file):
        print(f"Model file {model_file} not found! Using model_final.pth instead.")
        model_file = os.path.join(out_dir, "model_final.pth")
    
    if not os.path.exists(model_file):
        print(f"No trained model found in {out_dir}")
        return
    
    cfg.MODEL.WEIGHTS = model_file
    print(f"Evaluating model: {model_file}")
    
    try:
        evaluator = COCOEvaluator("tomato_test", cfg, False, output_dir=out_dir)
        val_loader = build_detection_test_loader(cfg, "tomato_test")
        model = MyTrainer.build_model(cfg)
        DefaultTrainer._load_checkpoint_to_model(model, model_file)
        results = inference_on_dataset(model, val_loader, evaluator)
        
        print("\nTest Evaluation Results:")
        print(json.dumps(results, indent=2))
        
        # 保存结果到文件
        results_file = os.path.join(out_dir, f"test_results_{model_path.replace('.pth', '')}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_file}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

def visualize_random_sample(n=3, model_path="model_best.pth", output_dir=None):
    """可视化随机样本的预测结果"""
    cfg = build_cfg(output_dir=output_dir)  # 使用默认配置
    out_dir = output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR
    model_file = os.path.join(out_dir, model_path)
    
    if not os.path.exists(model_file):
        print(f"Model file {model_file} not found! Using model_final.pth instead.")
        model_file = os.path.join(out_dir, "model_final.pth")
    
    if not os.path.exists(model_file):
        print(f"No trained model found in {out_dir}")
        return
    
    cfg.MODEL.WEIGHTS = model_file
    predictor = DefaultPredictor(cfg)
    
    try:
        metadata = MetadataCatalog.get("tomato_val")
    except:
        print("Warning: Could not get metadata for tomato_val")
        metadata = None

    if not os.path.exists(IMG_DIR):
        print(f"Image directory {IMG_DIR} does not exist!")
        return
    
    img_list = [f for f in os.listdir(IMG_DIR) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    
    if not img_list:
        print(f"No image files found in {IMG_DIR}")
        return
    
    random.shuffle(img_list)
    shown = 0

    print(f"Generating {n} visualization samples using {model_path}...")
    
    for file in img_list:
        try:
            img_path = os.path.join(IMG_DIR, file)
            im = cv2.imread(img_path)
            
            if im is None:
                print(f"Failed to load image: {img_path}")
                continue
                
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.2)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            save_path = os.path.join(out_dir, f"prediction_{shown}_{file}")
            cv2.imwrite(save_path, out.get_image()[:, :, ::-1])
            print(f"Saved: {save_path}")

            shown += 1
            if shown >= n:
                break
                
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

def get_dataset_info():
    """获取数据集信息以帮助设置正确的参数"""
    try:
        cfg = build_cfg()  # 使用默认配置
        train_loader = build_detection_train_loader(cfg)
        train_size = len(train_loader)
        
        print(f"\nDataset Information:")
        print(f"{'='*40}")
        print(f"Training dataset size: {train_size} images")
        print(f"Images per batch: {cfg.SOLVER.IMS_PER_BATCH}")
        print(f"Iterations per epoch: {train_size // cfg.SOLVER.IMS_PER_BATCH}")
        print(f"Total iterations for {MAX_EPOCHS} epochs: {(train_size // cfg.SOLVER.IMS_PER_BATCH) * MAX_EPOCHS}")
        
        return train_size
    except Exception as e:
        print(f"Could not determine dataset size: {e}")
        return None

def analyze_dataset_areas():
    """分析数据集中物体的面积分布"""
    import json
    import numpy as np
    
    print(f"\nAnalyzing object area distribution for 5K images...")
    print(f"{'='*60}")
    
    for split in ['train', 'val', 'test']:
        ann_file = os.path.join(ANN_DIR, f"{split}.json")
        if not os.path.exists(ann_file):
            print(f"Annotation file {ann_file} not found")
            continue
            
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # 创建图像ID到图像信息的映射
        image_info = {img['id']: img for img in data['images']}
        
        areas_original = []
        areas_scaled = []
        
        # 获取配置以了解缩放参数
        cfg = build_cfg()  # 使用默认配置
        min_size = min(cfg.INPUT.MIN_SIZE_TRAIN)
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        
        for ann in data['annotations']:
            # 原始面积
            if 'area' in ann:
                area = ann['area']
            else:
                bbox = ann.get('bbox', [0, 0, 0, 0])
                area = bbox[2] * bbox[3]
            areas_original.append(area)
            
            # 估算缩放后的面积
            img_id = ann['image_id']
            if img_id in image_info:
                img = image_info[img_id]
                orig_w, orig_h = img['width'], img['height']
                
                # 模拟Detectron2的缩放逻辑
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
        print(f"-" * 40)
        
        if len(data['images']) > 0:
            avg_width = np.mean([img['width'] for img in data['images']])
            avg_height = np.mean([img['height'] for img in data['images']])
            print(f"Average image size: {avg_width:.0f} x {avg_height:.0f}")
        
        print(f"Total objects: {len(areas_original)}")
        
        # 原始图像中的分布
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
        
        # 缩放后的分布
        print(f"\nAfter scaling to {min_size}-{max_size}:")
        small_scaled = np.sum(areas_scaled < 32**2)
        medium_scaled = np.sum((areas_scaled >= 32**2) & (areas_scaled < 96**2))
        large_scaled = np.sum(areas_scaled >= 96**2)
        
        print(f"  Small (<32²): {small_scaled} ({small_scaled/len(areas_scaled)*100:.1f}%)")
        print(f"  Medium (32²-96²): {medium_scaled} ({medium_scaled/len(areas_scaled)*100:.1f}%)")
        print(f"  Large (>96²): {large_scaled} ({large_scaled/len(areas_scaled)*100:.1f}%)")
        
        if small_scaled == 0:
            print(f"\n ! No small objects after scaling - APs metric will be -1")
        if medium_scaled == 0:
            print(f"! No medium objects after scaling - APm metric will be -1")
        if large_scaled == 0:
            print(f"! No large objects after scaling - APl metric will be -1")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detectron2 Training Script for Instance Segmentation")
    
    # 主要动作
    parser.add_argument("action", choices=["train", "eval", "vis", "info", "analyze"], 
                       help="Action to perform: train / eval / vis / info / analyze")
    
    # 训练相关参数
    parser.add_argument("--lr", type=float, default=None,
                       help="Base learning rate (default: 0.0001)")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Maximum number of epochs to train (default: 100)")
    parser.add_argument("--patience", type=int, default=None,
                       help="Early stopping patience in epochs (default: 10)")
    parser.add_argument("--model", type=str, default=None,
                       help="Model config file, e.g., 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml' (default: mask_rcnn_R_50_FPN_1x.yaml)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for saving models and results (default: ./r50x1)")
    
    # 评估和可视化相关参数
    parser.add_argument("--model-path", type=str, default="model_best.pth",
                       help="Model checkpoint file to use for eval/vis (default: model_best.pth)")
    parser.add_argument("--n", type=int, default=3,
                       help="Number of images to visualize (default: 3)")
    
    args = parser.parse_args()

    if args.action == "train":
        # 可用的模型配置示例
        print("\nAvailable model configs:")
        print("  - COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
        print("  - COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        print("  - COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        print("  - COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
        print("")
        
        train(model_config=args.model, 
              base_lr=args.lr, 
              max_epochs=args.epochs, 
              patience=args.patience,
              output_dir=args.output)
    
    elif args.action == "eval":
        evaluate(args.model_path, output_dir=args.output)
    
    elif args.action == "vis":
        visualize_random_sample(n=args.n, model_path=args.model_path, output_dir=args.output)
    
    elif args.action == "info":
        get_dataset_info()
    
    elif args.action == "analyze":
        analyze_dataset_areas()


# python seg.py train --model COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml --lr 0.00096 --epoch 100 --patience 5
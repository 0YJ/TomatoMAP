#!/usr/bin/env python3
"""
Segmentation model hooks and trainer for TomatoMAP-Seg with Detectron2
"""

import os
import json
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.events import get_event_storage

from utils.common import print_section


class BestModelHook(HookBase):
    
    def __init__(self, cfg, eval_period, patience=10):
        self.cfg = cfg.clone()
        self.eval_period = eval_period
        self.patience = patience
        self.best_score = 0
        self.best_metric_name = None
        self.best_epoch = -1
        self.epochs_without_improvement = 0
        self.should_stop = False
        self.history = []
        
    def get_valid_score(self, segm_results):
        # priority order for metrics
        priority_metrics = ["AP", "AP50", "AP75", "APm", "APl"]
        
        for metric in priority_metrics:
            value = segm_results.get(metric, -1)
            if value != -1:  # Only return valid values
                return metric, value
        
        return None, None
    
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final_iter = next_iter == self.trainer.max_iter
        
        if (next_iter % self.eval_period == 0 and not is_final_iter):
            current_epoch = (next_iter // self.eval_period)
            
            # perform evaluation
            results = self._do_eval()
            if results is None:
                print(f"Epoch {current_epoch}: Evaluation failed")
                return
            
            segm_results = results.get("segm", {})
            bbox_results = results.get("bbox", {})
            
            # get valid score
            metric_name, current_score = self.get_valid_score(segm_results)
            
            # print results
            print(f"\n{'='*60}")
            print(f"Epoch {current_epoch} Evaluation Results:")
            print(f"{'='*60}")
            
            # print bbox metrics
            print("\nBounding Box Metrics:")
            for key in ["AP", "AP50", "AP75", "APs", "APm", "APl"]:
                value = bbox_results.get(key, -1)
                if value != -1:
                    print(f"  {key}: {value:.4f} ✓")
                else:
                    print(f"  {key}: N/A")
            
            # print segmentation metrics
            print("\nSegmentation Metrics:")
            for key in ["AP", "AP50", "AP75", "APs", "APm", "APl"]:
                value = segm_results.get(key, -1)
                if value != -1:
                    print(f"  {key}: {value:.4f} ✓")
                else:
                    print(f"  {key}: N/A (no objects in this category)")
            
            # handle no valid metrics
            if metric_name is None:
                print("\n⚠️ Warning: No valid metrics available for evaluation!")
                print("This might happen if your dataset has no objects after augmentation.")
                # Try using bbox metrics as fallback
                metric_name, current_score = self.get_valid_score(bbox_results)
                if metric_name is not None:
                    print(f"Using bbox metric instead: {metric_name} = {current_score:.4f}")
                else:
                    return
            
            print(f"\nPrimary metric: {metric_name} = {current_score:.4f}")
            
            # record history
            self.history.append({
                'epoch': current_epoch,
                'metric': metric_name,
                'score': current_score,
                'all_metrics': {**segm_results, **{'bbox_' + k: v for k, v in bbox_results.items()}}
            })
            
            # check for improvement
            if current_score > self.best_score:
                improvement = current_score - self.best_score
                self.best_score = current_score
                self.best_metric_name = metric_name
                self.best_epoch = current_epoch
                self.epochs_without_improvement = 0
                
                # save model
                self.trainer.checkpointer.save("model_best")
                print(f"\n🎉 New best model saved!")
                print(f"   Score: {current_score:.4f} (↑{improvement:.4f})")
                
                # save evaluation results
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
            
            # early stop check
            if self.epochs_without_improvement >= self.patience:
                print(f"\n{'='*60}")
                print(f"EARLY STOPPING TRIGGERED")
                print(f"   Best {self.best_metric_name}: {self.best_score:.4f} at epoch {self.best_epoch}")
                print(f"   Total epochs trained: {current_epoch}")
                print(f"{'='*60}")
                self.should_stop = True
                # stop training
                self.trainer.storage._iter = self.trainer.max_iter
    
    def _do_eval(self):
        try:
            evaluator = COCOEvaluator(
                "tomato_val", self.cfg, False,
                output_dir=os.path.join(self.cfg.OUTPUT_DIR, "inference")
            )
            val_loader = build_detection_test_loader(self.cfg, "tomato_val")
            results = inference_on_dataset(self.trainer.model, val_loader, evaluator)
            return results
        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None


class SegmentationTrainer(DefaultTrainer):
    
    def __init__(self, cfg, patience=None):
        self.patience = patience if patience is not None else 5
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
        
        # calculate iterations per epoch
        try:
            train_loader = build_detection_train_loader(self.cfg)
            iters_per_epoch = len(train_loader) // self.cfg.SOLVER.IMS_PER_BATCH
            print(f"Iterations per epoch: {iters_per_epoch}")
        except:
            # use estimated value
            iters_per_epoch = 127
            print(f"Using estimated iterations per epoch: {iters_per_epoch}")
        
        # evaluate every 10 epochs
        eval_period = iters_per_epoch * 10
        
        # add best model hook
        best_model_hook = BestModelHook(self.cfg, eval_period, self.patience)
        hooks.append(best_model_hook)
        
        # store hook reference
        self.best_model_hook = best_model_hook
        
        return hooks
    
    def run_step(self):
        super().run_step()
        
        # check for early stopping
        if hasattr(self, 'best_model_hook') and self.best_model_hook.should_stop:
            print("Early stopping condition met. Stopping training...")
            self.storage._iter = self.max_iter
    
    def train(self):
        super().train()
        
        # training summary
        if hasattr(self, 'best_model_hook'):
            print_section("Training Summary")
            if self.best_model_hook.best_score > 0:
                print(f"Best {self.best_model_hook.best_metric_name}: {self.best_model_hook.best_score:.4f}")
                print(f"Best epoch: {self.best_model_hook.best_epoch}")
                print(f"Best model saved as: model_best.pth")
            else:
                print("No valid metrics were found during training.")
            
            # save training history
            history_file = os.path.join(self.cfg.OUTPUT_DIR, "training_history.json")
            with open(history_file, 'w') as f:
                json.dump(self.best_model_hook.history, f, indent=2)
            print(f"Training history saved to: {history_file}")

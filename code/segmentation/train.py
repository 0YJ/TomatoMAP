from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
import os

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        from detectron2.data import MetadataCatalog
        from detectron2.evaluation import COCOEvaluator

        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(
            dataset_name=dataset_name,
            cfg=cfg,
            distributed=False,
            output_dir=output_folder,
            use_fast_impl=True,
            tasks=("bbox", "segm"),
            metadata=MetadataCatalog.get(dataset_name)
        )

register_coco_instances("tomato_train", {}, "cocoOut/train.json", "task1")
register_coco_instances("tomato_val", {}, "cocoOut/val.json", "task1")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("tomato_train",)
cfg.DATASETS.TEST = ("tomato_val",)
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 100
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.OUTPUT_DIR = "./output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

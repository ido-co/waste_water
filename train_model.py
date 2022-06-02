# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

l = setup_logger(output="./my_log_file.log")

# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator

from detectron2.data.datasets import register_coco_instances

from utils import MyTrainer

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

register_coco_instances("ww_train", {}, "Waste_Water_low_res/ww_data_train.json", "Waste_Water_low_res/train")
register_coco_instances("ww_eval", {}, "Waste_Water_low_res/ww_data_eval.json", "Waste_Water_low_res/train")
register_coco_instances("ww_test", {}, "Waste_Water_low_res/ww_data_test.json", "Waste_Water_low_res/train")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))
cfg.DATASETS.TRAIN = ("ww_train",)
cfg.DATASETS.TEST = ("ww_eval",)  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.002
cfg.SOLVER.MAX_ITER = 10  # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # 3 classes (data, fig, hazelnut)
cfg.TEST.EVAL_PERIOD = 100

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

evaluator = COCOEvaluator("ww_test", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "ww_test")
out = inference_on_dataset(trainer.model, val_loader, evaluator)



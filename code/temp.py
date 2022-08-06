# from contextlib import redirect_stdout
#
# with open('../help.txt', 'w') as f:
#     with redirect_stdout(f):
#         help(pow)
import argparse
import os
import pyodi
import os
from copy import deepcopy

from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.structures.boxes import BoxMode

setup_logger()

# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
import re
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
import json

# args = parser.parse_args()
# print((args))

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.detection_utils import annotations_to_instances

# evaluator = COCOEvaluator("ww_test", cfg, False, output_dir="./output/")
# val_loader = build_detection_test_loader(cfg, "ww_test")
# inference_on_dataset(trainer.model, val_loader, evaluator)
#
# for m in tmp2['instances'] :
#     m["bbox_mode"] = bbox_mode
import torch
from detectron2.evaluation import DatasetEvaluator

# register_coco_instances("ww_test", {}, "data/file_out/ww_data_test.json", "data/file_out/")
# t = torch.load("data/instances_predictions .pth")
# dataset_dicts = DatasetCatalog.get("ww_test")
# tmp1 = [img for img in dataset_dicts if img["image_id"] == 32][0]
# tmp2 = t[6]
# bbox_mode = BoxMode.XYWH_ABS
# for m in tmp2['instances']:
#     m["bbox_mode"] = bbox_mode
# scores = [d["score"] for d in tmp2['instances']]
# ins = annotations_to_instances(annos=tmp2['instances'], image_size=(1920, 2560))
# ins._fields["pred_boxes"] = ins._fields["gt_boxes"]
# ins._fields["scores"] = scores
# ins._fields["pred_classes"] = ins._fields["gt_classes"]
# evaluator = DatasetEvaluator()
#
register_coco_instances("ww_train", {}, "../data/file_out/ww_data_train.json", "data/file_out/")
register_coco_instances("ww_eval", {}, "../data/file_out/ww_data_eval.json", "data/file_out/")
register_coco_instances("ww_test", {}, "../data/file_out/ww_data_test.json", "data/file_out/")
ww_test = DatasetCatalog.get("ww_test")
ww_eval = DatasetCatalog.get("ww_eval")
ww_train = DatasetCatalog.get("ww_train")
image_id_to_filename = {r["image_id"]: r["file_name"] for r in ww_test + ww_train + ww_eval}
image_id_to_dataset_obj = {r["image_id"]: r for r in ww_test + ww_train + ww_eval}
waste_water_metadata = MetadataCatalog.get("ww_test")
bbox_mode = BoxMode.XYWH_ABS


def visualize_by_dict(dictu,filename = "tmp_name"):
    return visualize_by_image_id(dictu["image_id"], dictu, dirname="tmp_dir",filename =filename)


def visualize_by_image_id(img_id, pred, dirname="tmp_dir",filename = "tmp_name"):
    img = cv2.imread("../" + image_id_to_filename[img_id])
    if img is None:
        return
    visualizer = Visualizer(img[:, :, ::-1], metadata=waste_water_metadata, scale=1)
    obj = deepcopy(image_id_to_dataset_obj[img_id])
    pred_d_d = deepcopy(obj)
    pred_d_d["annotations"] = [pred]
    pred_d_d["annotations"][0]['bbox_mode'] = bbox_mode
    pred_d_d["annotations"][0]["bbox"] = pred["bbox"]
    pred_d_d["annotations"][0]["category_id"] = pred["category_id"]
    vis = visualizer.draw_dataset_dict(pred_d_d)
    cv2.imwrite(f"../{dirname}/{filename}.png", vis.get_image()[:, :, ::-1])
    return vis.get_image()

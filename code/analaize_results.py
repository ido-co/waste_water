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


def safe_makdir(path):
    if os.path.exists(path):
        return
    else:
        os.mkdir(path)


image_path = r"C:\Users\Administrator\PycharmProjects\ml_workshop\wastwatwer_data\Waste_Water_low_res\train"

register_coco_instances("ww_train", {}, "../wastwatwer_data/Waste_Water_low_res/ww_data_train.json",
                        image_path)
register_coco_instances("ww_eval", {}, "../wastwatwer_data/Waste_Water_low_res/ww_data_eval.json",
                        image_path)
register_coco_instances("ww_test", {}, "../wastwatwer_data/Waste_Water_low_res/ww_data_test.json",
                        image_path)

# register_coco_instances("ww_train", {}, "../wastwatwer_data/Waste_Water_low_res/ww_data_train.json",
#                         "wastwater_data/Waste_Water_low_res/train")
# register_coco_instances("ww_eval", {}, "../wastwatwer_data/Waste_Water_low_res/ww_data_eval.json",
#                         "wastwatwer_data/Waste_Water_low_res/train")
# register_coco_instances("ww_test", {}, "../wastwatwer_data/Waste_Water_low_res/ww_data_test.json",
#                         "wastwater_data/Waste_Water_low_res/train")

res = json.load(open("../wastwatwer_data/ww_output/output_5_16/coco_instances_results.json", 'r'))
inf = json.load(open("../wastwatwer_data/ww_output/output_5_16/inference/coco_instances_results.json", 'r'))
# len(dataset_dicts[0])
waste_water_metadata = MetadataCatalog.get("ww_test")
ww_test = DatasetCatalog.get("ww_test")
ww_eval = DatasetCatalog.get("ww_eval")
ww_train = DatasetCatalog.get("ww_train")

image_id_to_filename = {r["image_id"]: r["file_name"] for r in ww_test + ww_train + ww_eval}
image_id_to_dataset_obj = {r["image_id"]: r for r in ww_test + ww_train + ww_eval}

scores = np.array([r["score"] for r in res])
wrong = [r for r in res if r["score"] < 0.5]

from collections import defaultdict

# collect all
preds = defaultdict(list)
confdence_threshhold = 0.7
iou_threshhold = 0.7
iou = 0


def calc_iou(A, B):
    boxA = [A[0], A[1], A[0] + A[2], A[3] + A[3]]
    boxB = [B[0], B[1], B[0] + B[2], B[3] + B[3]]
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def get_iou(pred, truth):
    if truth["annotations"]:
        iou_scores = [calc_iou(box["bbox"], pred["bbox"]) for box in truth["annotations"]]
        return max(iou_scores)
    return 0


count = 0
count_score = 0
for r in res:
    if r["score"] > confdence_threshhold:
        count_score += 1
    iou = get_iou(r, image_id_to_dataset_obj[r["image_id"]])
    if r["score"] > confdence_threshhold and \
            iou_threshhold > iou:
        preds[r["image_id"]].append((iou, r))
        count += 1
bbox_mode = BoxMode.XYWH_ABS
# save the preds
# safe_makdir("../result_images")
offset_y = 30
offset_x = 150
for image_id in (preds.keys()):
    tru = image_id_to_dataset_obj[image_id]
    name = re.findall("[a-zA-Z0-9]+_\d+_\d+_\d+_[a-zA-Z0-9]+_\d+", tru['file_name'])
    safe_makdir(f"../result_images/{name}")
    img = cv2.imread(tru["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=waste_water_metadata, scale=1)
    vis = visualizer.draw_dataset_dict(tru)
    cv2.imwrite(f"../result_images/{name}/ground_truth.png", vis.get_image()[:, :, ::-1])
    for iou, pred in preds[image_id]:
        pred_d_d = deepcopy(tru)
        pred_d_d["annotations"] = [pred]
        pred_d_d["annotations"][0]['bbox_mode'] = bbox_mode
        pred_d_d["annotations"][0]["bbox"] = pred["bbox"]
        pred_d_d["annotations"][0]["category_id"] = pred["category_id"]
        visualizer = Visualizer(img[:, :, ::-1], metadata=waste_water_metadata, scale=1)
        visualizer.draw_dataset_dict(pred_d_d)
        loc = (pred_d_d["annotations"][0]["bbox"][0] + offset_x, pred_d_d["annotations"][0]["bbox"][1] + offset_y)
        vis = visualizer.draw_text(f"%{pred['score']}", loc, font_size=20)
        cv2.imwrite(f"../result_images/{name}/iou_{iou}.png", vis.get_image()[:, :, ::-1])
        # import matplotlib.pyplot as plt
        # plt.imshow(vis.get_image()[:, :, ::-1])
        # plt.show()

#
for i in list(preds.keys())[:3]:
    d = image_id_to_dataset_obj[i]
    for dt in preds[i]:
        my_d = deepcopy(d)
        my_d["annotations"][0]["bbox"] = dt["bbox"]
        my_d["annotations"][0]["category_id"] = dt["category_id"]
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=waste_water_metadata, scale=0.1)
        # vis = visualizer.draw_dataset_dict(my_d)
        # cv2.imshow(vis.get_image()[:, :, ::-1])
        # print(my_d)
        # print(f"score ={dt['score']}")




import matplotlib.pyplot as plt
plt.imshow(img[:, :, ::-1])
plt.imshow(vis.get_image()[:, :, ::-1])
plt.show()



# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup
# Setup detectron2 logger
import torch

if __name__ == '__main__':
    import sys

    sys.path.append("/home/yandex/MLW2022/idoc/detectron2/detectron2/")
import detectron2
from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads import Res5ROIHeads, select_foreground_proposals
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

if __name__ == '__main__':
    register_coco_instances("ww_train", {}, "../data/file_out/ww_data_train.json", "../data/file_out/")
    register_coco_instances("ww_eval", {}, "../data/file_out/ww_data_eval.json", "../data/file_out/")
    register_coco_instances("ww_test", {}, "../data/file_out/ww_data_test.json", "../data/file_out/")


    # makin my oun roi head
    @ROI_HEADS_REGISTRY.register()
    class My_Res5ROIHeads(Res5ROIHeads):
        # def __init__(
        #         self,
        #         *,
        #         in_features: List[str],
        #         pooler: ROIPooler,
        #         res5: nn.Module,
        #         box_predictor: nn.Module,
        #         mask_head: Optional[nn.Module] = None,
        #         **kwargs,
        # ):
        #     super().__init__(in_features,
        #                      pooler,
        #                      res5,
        #                      box_predictor,
        #                      mask_head, kwargs)
        def forward(
                self,
                images,
                features,
                proposals,
                targets=None,
        ):
            """
            See :meth:`ROIHeads.forward`.
            """
            del images

            if self.training:
                assert targets
                proposals = self.label_and_sample_proposals(proposals, targets)
            del targets

            proposal_boxes = [x.proposal_boxes for x in proposals]
            box_features = self._shared_roi_transform(
                [features[f] for f in self.in_features], proposal_boxes
            )
            predictions = self.box_predictor(box_features.mean(dim=[2, 3]))

            if self.training:
                del features
                losses = self.box_predictor.losses(predictions, proposals)
                if self.mask_on:
                    proposals, fg_selection_masks = select_foreground_proposals(
                        proposals, self.num_classes
                    )
                    # Since the ROI feature transform is shared between boxes and masks,
                    # we don't need to recompute features. The mask loss is only defined
                    # on foreground proposals, so we need to select out the foreground
                    # features.
                    mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                    del box_features
                    losses.update(self.mask_head(mask_features, proposals))
                return [], losses
            else:
                pred_instances, _ = self.box_predictor.inference(predictions, proposals)
                pred_instances = self.forward_with_given_boxes(features, pred_instances)
            # for editing the proposals boxes  proposals[0]._fields["proposal_boxes"].tensor
            orig_res = pred_instances, {}
            # orig_res = super().forward(images, features, proposals, targets)
            if not self.training:
                # print("=" * 9 + "my module" + "=" * 9)
                if self.filter_flag:
                    ins = pred_instances[0]
                    # ins[ins._fields["pred_classes"] == 2]._fields["pred_boxes"]
                    keep = torch.ones(len(ins), dtype=bool,device=ins._fields["pred_classes"].device)
                    iou_mat = detectron2.structures.pairwise_iou(
                        ins[ins._fields["pred_classes"] == self.OPEN_FLOCK]._fields["pred_boxes"],
                        ins[ins._fields["pred_classes"] == self.SPHERICAL_FLOCK]._fields["pred_boxes"])
                    keep[ins._fields["pred_classes"] == self.OPEN_FLOCK] = iou_mat.max(dim=1).values > self.filter_threshold
                    ins._fields["pred_classes"] = ins._fields["pred_classes"][keep]
                    ins._fields["scores"] = ins._fields["scores"][keep]
                    ins._fields["pred_boxes"] = ins._fields["pred_boxes"][keep]
                    # detectron2.structures.boxes.pairwise_intersection(
                    #     ins[ins._fields["pred_classes"] == 2]._fields["pred_boxes"],
                    #     ins[ins._fields["pred_classes"] == 2]._fields["pred_boxes"])
                # get the predicted boxes with pred_instances[0]._fields["pred_boxes"].tensor
                # get the class predicted for that box with pred_instances[0]._fields["pred_classes"]
                # todo need to make sure that removing a sample from the prediction don't break anything

                # print(orig_res)
                # print("=" * 22)
            return orig_res


    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))
    cfg.DATASETS.TRAIN = ("ww_train",)
    cfg.DATASETS.TEST = ("ww_eval",)  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.002
    cfg.SOLVER.MAX_ITER = 1  # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.NAME = "My_Res5ROIHeads"
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # 3 classes (data, fig, hazelnut)
    cfg.TEST.EVAL_PERIOD = 100

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.model.roi_heads.filter_threshold = 0.13
    trainer.model.roi_heads.SPHERICAL_FLOCK = 4
    trainer.model.roi_heads.OPEN_FLOCK = 2
    trainer.model.roi_heads.filter_flag = True
    trainer.train()

    evaluator = COCOEvaluator("ww_test", cfg, False, output_dir="../output/")
    val_loader = build_detection_test_loader(cfg, "ww_test")
    out = inference_on_dataset(trainer.model, val_loader, evaluator)

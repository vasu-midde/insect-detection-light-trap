"""Faster R-CNN ResNet50-FPN v2 (COCO-pretrained) set up for one class, without hidden resizing.

mode="tiled": input is a tile (cfg tile x tile) cut from the frame shrunk by tile_scale
mode="down" : input is the whole frame shrunk by cfg down_factor (2560 x 1440)
"""
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def input_size(cfg, mode):
    if mode == "tiled":
        return cfg["tile"], cfg["tile"]
    W, H = cfg["frame_size"]
    return H // cfg["down_factor"], W // cfg["down_factor"]          # (short side, long side)


def build_model(cfg, mode="tiled", pretrained=True):
    weights = "DEFAULT" if pretrained else None
    m = fasterrcnn_resnet50_fpn_v2(weights=weights, weights_backbone=None,
                                   box_detections_per_img=300, box_score_thresh=0.01)
    # 2 classes: background + insect
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)

    # anchors from the measured insect sizes (same number per location, so pretrained RPN weights still fit)
    f = cfg["tile_scale"] if mode == "tiled" else cfg["down_factor"]
    sizes = [max(16, int(s / f)) for s in cfg["anchor_sizes"]]
    m.rpn.anchor_generator = AnchorGenerator(sizes=tuple((int(s),) for s in sizes),
                                             aspect_ratios=(tuple(cfg["anchor_ratios"]),) * len(sizes))

    # switch off torchvision's default resize to 800 px: images go in at the size we give them
    short, long_ = input_size(cfg, mode)
    m.transform.min_size = (short,)
    m.transform.max_size = long_
    return m


def load_model(path, cfg, mode, device):
    m = build_model(cfg, mode, pretrained=False)
    state = torch.load(path, map_location="cpu")
    m.load_state_dict(state["model"] if "model" in state else state)
    return m.to(device).eval()

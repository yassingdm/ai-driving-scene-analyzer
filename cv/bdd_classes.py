"""BDD100K target classes and COCO-to-BDD mapping."""

from __future__ import annotations


BDD_CLASSES = [
    "pedestrian",
    "cyclist",
    "motorcycle",
    "truck",
    "bus",
    "car",
    "traffic_light",
    "traffic_sign",
    "rider",
    "train",
]

# COCO label -> BDD100K label (for zero-shot / pretrain outputs).
COCO_TO_BDD = {
    "person": "pedestrian",
    "bicycle": "cyclist",
    "motorcycle": "motorcycle",
    "truck": "truck",
    "bus": "bus",
    "car": "car",
    "traffic light": "traffic_light",
    "stop sign": "traffic_sign",
    "train": "train",
}

# BDD100K label -> COCO label (if needed for reverse mapping).
BDD_TO_COCO = {v: k for k, v in COCO_TO_BDD.items()}

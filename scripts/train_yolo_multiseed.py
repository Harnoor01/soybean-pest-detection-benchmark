"""
train_yolo_multiseed.py

Drop-in replacement for train_yolo.py that accepts a --seed argument.
The seed controls BOTH:
  1. The data.yaml path (pointing to the seed-specific split)
  2. The YOLO/RT-DETR training seed (for reproducible weight init, dropout, etc.)

Usage:
    python scripts/train_yolo_multiseed.py \
        --model yolov8s \
        --exp_name E1_yolov8s_seed0 \
        --aug yolo_default \
        --seed 0 \
        --device 0

The --data argument is auto-derived from --seed unless --data is explicitly passed.
"""

import argparse
import copy
import os
import random

import numpy as np

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",    required=True,
                    choices=["yolov8s", "rtdetr-l", "yolo26s"])
parser.add_argument("--exp_name", required=True,
                    help="Experiment name (used as YOLO project/name)")
parser.add_argument("--aug",      required=True,
                    help="Augmentation condition key from ABLATION_CONDITIONS")
parser.add_argument("--seed",     type=int, default=42,
                    help="Seed for both the data split (selects data.yaml) "
                         "and training RNG")
parser.add_argument("--data",     default=None,
                    help="Explicit path to data.yaml (overrides auto-derive)")
parser.add_argument("--device",   default="0",
                    help="CUDA device index (default: 0)")
args = parser.parse_args()

SEED = args.seed

# ── Seed all RNGs ─────────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except ImportError:
    pass

# ── Configs (mirrors experiment_config.py) ────────────────────────────────────
_TRAIN_SHARED = dict(
    imgsz=640,
    epochs=100,
    batch=16,
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    optimizer="SGD",
    patience=50,
    workers=8,
    seed=SEED,          # <-- overridden by CLI --seed
    deterministic=True,
    pretrained=True,
    amp=True,
    close_mosaic=10,
)

_YOLO26_TRAIN_SHARED = {**_TRAIN_SHARED, "optimizer": "auto"}

_RTDETR_TRAIN_SHARED = dict(
    imgsz=640,
    epochs=100,
    batch=16,
    lr0=0.0001,
    lrf=0.01,
    weight_decay=0.0001,
    optimizer="AdamW",
    warmup_epochs=0,
    patience=50,
    seed=SEED,          # <-- overridden by CLI --seed
    deterministic=True,
    pretrained=True,
    amp=True,
    close_mosaic=0,
)

# Augmentation conditions (keep in sync with experiment_config.py)
_AUG_YOLO_DEFAULT = dict(
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    translate=0.1, scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    close_mosaic=10,
)

ABLATION_CONDITIONS = {
    "no_aug": dict(
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        translate=0.0, scale=0.0,
        fliplr=0.0,
        mosaic=0.0,
        close_mosaic=0,
    ),
    "yolo_default": _AUG_YOLO_DEFAULT,
    "mosaic_only": dict(
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        translate=0.0, scale=0.0,
        fliplr=0.0,
        mosaic=1.0,
        close_mosaic=10,
    ),
    "yolo_default_bbox_cp": {
        **_AUG_YOLO_DEFAULT,
        "copy_paste": 0.3,
        "copy_paste_mode": "flip",
    },
    "close_mosaic_0":  {**_AUG_YOLO_DEFAULT, "close_mosaic": 0},
    "close_mosaic_10": {**_AUG_YOLO_DEFAULT, "close_mosaic": 10},
    "close_mosaic_50": {**_AUG_YOLO_DEFAULT, "close_mosaic": 50},
}

# ── Model registry ────────────────────────────────────────────────────────────
# (weights_file, base_train_cfg, model_type)
MODEL_REGISTRY = {
    "yolov8s":  ("yolov8s.pt",  _TRAIN_SHARED,       "YOLO"),
    "rtdetr-l": ("rtdetr-l.pt", _RTDETR_TRAIN_SHARED, "RTDETR"),
    "yolo26s":  ("yolo26s.pt",  _YOLO26_TRAIN_SHARED, "YOLO"),
}

# ── Data path ─────────────────────────────────────────────────────────────────
BASE = ("/project/def-grandha8-ab/harnoor1/pest_project_EPA/"
        "soybean_detection_project")

if args.data:
    DATA_YAML = args.data
else:
    # Auto-derive from seed — points to the seed-specific split
    DATA_YAML = (f"{BASE}/soybean_yolo_splits/seed{SEED}/data.yaml")

print(f"[seed={SEED}] Using data: {DATA_YAML}")

# ── Build training kwargs ─────────────────────────────────────────────────────
weights_file, base_cfg, model_type = MODEL_REGISTRY[args.model]

# Deep copy so we never mutate the global config
train_kw = copy.deepcopy(base_cfg)
train_kw["seed"] = SEED  # ensure CLI seed wins

# Apply augmentation
aug_kw = copy.deepcopy(ABLATION_CONDITIONS[args.aug])
train_kw.update(aug_kw)

# RT-DETR doesn't support mosaic-family augmentations
if model_type == "RTDETR":
    for key in ["mosaic", "copy_paste", "mixup", "close_mosaic", "erasing"]:
        train_kw.pop(key, None)

# Inject data and project/name
train_kw["data"]    = DATA_YAML
train_kw["project"] = f"{BASE}/results"
train_kw["name"]    = args.exp_name
train_kw["device"]  = args.device

# ── Train ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Model    : {args.model}  ({weights_file})")
print(f"Exp name : {args.exp_name}")
print(f"Aug      : {args.aug}")
print(f"Seed     : {SEED}")
print(f"Data     : {DATA_YAML}")
print(f"{'='*60}\n")

if model_type == "YOLO":
    from ultralytics import YOLO
    model = YOLO(weights_file)
    model.train(**train_kw)
elif model_type == "RTDETR":
    from ultralytics import RTDETR
    model = RTDETR(weights_file)
    model.train(**train_kw)
else:
    raise ValueError(f"Unknown model_type: {model_type}")

print(f"\n[seed={SEED}] Training complete. Results in: {train_kw['project']}")
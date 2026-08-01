#!/usr/bin/env python3
"""
train_yolo.py
=============
Trains a YOLO model (YOLOv8s or YOLO26s) for soybean pest detection.

Models supported
----------------
  yolov8s  — Ultralytics YOLOv8 small  (anchor-free, DFL, NMS, SGD)
  yolo26s  — Ultralytics YOLO26 small  (NMS-free, no DFL, MuSGD, STAL)

Usage
-----
  # E1 — architectural comparison:
  python train_yolo.py --model yolov8s  --exp_name E1_yolov8s
  python train_yolo.py --model yolo26s  --exp_name E1_yolo26s

  # E2 — augmentation ablation (anchored on yolov8s):
  python train_yolo.py --model yolov8s --exp_name E2_no_aug       --aug no_aug
  python train_yolo.py --model yolov8s --exp_name E2_yolo_default --aug yolo_default
  python train_yolo.py --model yolov8s --exp_name E2_cp_only      --aug copy_paste_only
  python train_yolo.py --model yolov8s --exp_name E2_default_cp   --aug yolo_default_cp

Notes
-----
  YOLO26s uses optimizer="auto" (MuSGD) from experiment_config.YOLO26_TRAIN_SHARED.
  YOLOv8s uses optimizer="SGD" from experiment_config.TRAIN_SHARED.
  All other hyperparameters are identical for a fair architectural comparison.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── make project root importable ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.experiment_config import (
    DATASET, TRAIN_SHARED, YOLO26_TRAIN_SHARED, RTDETR_TRAIN_SHARED,
    ABLATION_CONDITIONS, RESULTS_ROOT, SEED
)

# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────
# Maps CLI model name → (weights file, base config, ultralytics class name)
#
# "YOLO"  → from ultralytics import YOLO   (YOLOv8s, YOLO26s)
# "RTDETR"→ from ultralytics import RTDETR (RT-DETR-L)
#
# WHY three entries with different configs:
#   YOLOv8s  : optimizer=SGD,   lr0=0.01   — CNN one-stage established baseline
#   RT-DETR-L: optimizer=AdamW, lr0=0.0001 — transformer, different convergence
#   YOLO26s  : optimizer=auto,  lr0=0.01   — CNN one-stage cutting-edge
#
# Each uses its published/recommended defaults so we evaluate each architecture
# as it was designed — not artificially constrained to a shared optimizer that
# would disadvantage transformer or next-gen models.

MODEL_REGISTRY = {
    "yolov8s":  ("yolov8s.pt",   TRAIN_SHARED,        "YOLO"),
    "rtdetr-l": ("rtdetr-l.pt",  RTDETR_TRAIN_SHARED, "RTDETR"),
    "yolo26s":  ("yolo26s.pt",   YOLO26_TRAIN_SHARED, "YOLO"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = SEED) -> None:
    """Set all random seeds for full reproducibility.

    Note: YOLO internally calls this too when `deterministic=True`, but we set
    it here explicitly so Faster R-CNN experiments use the same function.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False   # disables auto-tuning → deterministic
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train detection model on InsectBase soybean pest dataset")
    p.add_argument("--model",    type=str, default="yolov8s",
                   choices=list(MODEL_REGISTRY.keys()),
                   help="Model to train: yolov8s | rtdetr-l | yolo26s")
    p.add_argument("--exp_name", type=str, required=True,
                   help="Experiment name; results saved to results/<exp_name>/")
    p.add_argument("--aug",      type=str, default="yolo_default",
                   choices=list(ABLATION_CONDITIONS.keys()),
                   help="Augmentation condition (default: yolo_default)")
    p.add_argument("--data",     type=str,
                   default=str(PROJECT_ROOT / "data" / "data.yaml"),
                   help="Path to data.yaml")
    p.add_argument("--resume",   action="store_true",
                   help="Resume training from last checkpoint")
    p.add_argument("--device",   type=str, default="0",
                   help="CUDA device id(s), e.g. '0' or '0,1'")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    set_seed(SEED)

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed.  Run: pip install ultralytics")

    # ── output directory ──────────────────────────────────────────────────────
    out_dir = RESULTS_ROOT / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── select config and class for this model ────────────────────────────────
    _, base_cfg, model_class = MODEL_REGISTRY[args.model]

    # ── build training kwargs ─────────────────────────────────────────────────
    aug_cfg  = ABLATION_CONDITIONS[args.aug]
    train_kw = {
        **base_cfg,          # model-specific base config
        **aug_cfg,           # augmentation condition overrides
        "data":    args.data,
        "project": str(RESULTS_ROOT),
        "name":    args.exp_name,
        "device":  args.device,
        "exist_ok": True,
    }

    # RT-DETR does not use YOLO-specific augmentation keys — remove them
    # to avoid Ultralytics warnings.  The compatible subset (flips, HSV) is kept.
    if model_class == "RTDETR":
        for key in ("mosaic", "copy_paste", "mixup", "close_mosaic", "erasing"):
            train_kw.pop(key, None)

    # ── save experiment metadata ──────────────────────────────────────────────
    meta = {
        "model":         args.model,
        "model_class":   model_class,
        "exp_name":      args.exp_name,
        "aug_condition": args.aug,
        "seed":          SEED,
        "data_yaml":     args.data,
        "train_kwargs":  train_kw,
        "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "experiment_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Experiment : {args.exp_name}")
    print(f"  Model      : {args.model}")
    print(f"  Aug cond   : {args.aug}")
    print(f"  Results    : {out_dir}")
    print(f"{'='*60}\n")

    # ── load model ────────────────────────────────────────────────────────────
    # YOLO("<variant>.pt") downloads COCO-pretrained weights on first run.
    # On Narval (no internet on compute nodes), pre-download on a login node:
    #   python -c "from ultralytics import YOLO; YOLO('yolov8s.pt'); YOLO('yolo26s.pt')"
    # then the weights are cached in ~/.config/Ultralytics/
    weights_file, _, model_class = MODEL_REGISTRY[args.model]
    model_path = weights_file
    if args.resume:
        model_path = str(out_dir / "weights" / "last.pt")
        print(f"Resuming from: {model_path}")

    print(f"  Model class: {model_class}")
    print(f"  Optimizer  : {base_cfg.get('optimizer', 'auto')}")

    # Load the correct Ultralytics class
    if model_class == "RTDETR":
        from ultralytics import RTDETR
        model = RTDETR(model_path)
    else:
        model = YOLO(model_path)

    # ── train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    results = model.train(**train_kw)
    elapsed = time.time() - t0

    print(f"\nTraining completed in {elapsed/3600:.2f} h")
    print(f"Best weights: {out_dir}/weights/best.pt")

    # ── save training summary ─────────────────────────────────────────────────
    summary = {
        "training_time_seconds": elapsed,
        "best_map50":    float(results.results_dict.get("metrics/mAP50(B)",   -1)),
        "best_map50_95": float(results.results_dict.get("metrics/mAP50-95(B)",-1)),
        "best_precision":float(results.results_dict.get("metrics/precision(B)",-1)),
        "best_recall":   float(results.results_dict.get("metrics/recall(B)",  -1)),
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nVal mAP@0.5      : {summary['best_map50']:.4f}")
    print(f"Val mAP@0.5:0.95 : {summary['best_map50_95']:.4f}")


if __name__ == "__main__":
    main()
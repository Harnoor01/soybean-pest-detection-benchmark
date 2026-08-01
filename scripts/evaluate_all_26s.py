#!/usr/bin/env python3
"""
evaluate_all.py
===============
Evaluates all trained models on the held-out test set.

Produces:
  - Per-model JSON with mAP@0.5, mAP@0.5:0.95, per-class AP50
  - Consolidated results/eval_summary.json for table generation

Usage
-----
  # Evaluate a YOLOv8 experiment:
  python evaluate_all.py --model_type yolo \
      --weights results/E1_yolov8s/weights/best.pt \
      --exp_name E1_yolov8s

  # Evaluate Faster R-CNN:
  python evaluate_all.py --model_type fasterrcnn \
      --weights results/E2_fasterrcnn/best.pt \
      --exp_name E2_fasterrcnn

  # Evaluate all experiments at once:
  python evaluate_all.py --eval_all

Notes
-----
  - Uses conf_thres=0.001 (not 0.25) for correct mAP sweep.
  - IoU threshold for NMS: 0.6 (see EVAL config).
  - mAP computed via pycocotools COCOeval for standardised results.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.experiment_config import DATASET, EVAL, RESULTS_ROOT, SEED


# ─────────────────────────────────────────────────────────────────────────────
# COCO-format helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_coco_gt_from_yolo(
    images_dir: Path,
    labels_dir: Path,
    class_names: List[str],
) -> dict:
    """
    Build a pycocotools-compatible ground-truth dictionary from YOLO-format
    labels.  Returns the dict (not a COCO object) so the caller can also
    create a COCO object via pycocotools.coco.COCO(dict).

    YOLO label format (per line):
        class_id  cx_norm  cy_norm  w_norm  h_norm
    All values normalised to [0, 1] relative to image dimensions.
    """
    from PIL import Image as PILImage

    categories = [{"id": i + 1, "name": n} for i, n in enumerate(class_names)]
    coco_gt = {"images": [], "annotations": [], "categories": categories}

    img_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    ann_id = 1

    for img_id, img_path in enumerate(img_paths, start=1):
        with PILImage.open(img_path) as im:
            W, H = im.size

        coco_gt["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": W,
            "height": H,
        })

        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                # convert normalised xywh → absolute xyxy
                x1 = (cx - bw / 2) * W
                y1 = (cy - bh / 2) * H
                abs_w = bw * W
                abs_h = bh * H
                coco_gt["annotations"].append({
                    "id":          ann_id,
                    "image_id":    img_id,
                    "category_id": cls_id + 1,   # 1-indexed
                    "bbox":        [x1, y1, abs_w, abs_h],
                    "area":        abs_w * abs_h,
                    "iscrowd":     0,
                })
                ann_id += 1

    return coco_gt, {img["file_name"]: img["id"] for img in coco_gt["images"]}


def run_coco_eval(
    coco_gt_dict: dict,
    dt_list: List[dict],
    class_names: List[str],
) -> dict:
    """
    Run pycocotools COCOeval and return a results dict.

    dt_list items:
        {"image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float}
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import io, contextlib

    # Load GT
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()

    if not dt_list:
        print("  [WARN] No detections produced — all metrics = 0.")
        n = len(class_names)
        return {
            "mAP50": 0.0, "mAP50_95": 0.0,
            "per_class_AP50": {n: 0.0 for n in class_names}
        }

    coco_dt = coco_gt.loadRes(dt_list)

    # Overall eval
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.iouThrs = np.array(EVAL["iou_map_thres"])
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
    evaluator.summarize()

    mAP50    = float(evaluator.stats[1])   # AP @ IoU=0.50
    mAP50_95 = float(evaluator.stats[0])   # AP @ IoU=0.50:0.95

    # Per-class AP@0.50
    per_class_AP50: Dict[str, float] = {}
    iou50_idx = np.where(evaluator.params.iouThrs == 0.50)[0][0]

    for cat_idx, cat_id in enumerate(evaluator.params.catIds):
        # precision shape: [T, R, K, A, M]  T=IoU, R=recall, K=cat, A=area, M=maxDet
        prec = evaluator.eval["precision"][iou50_idx, :, cat_idx, 0, 2]  # area=all, maxDet=100
        prec = prec[prec > -1]
        ap = float(prec.mean()) if prec.size > 0 else 0.0
        cat_name = class_names[cat_id - 1]
        per_class_AP50[cat_name] = ap

    return {
        "mAP50":          mAP50,
        "mAP50_95":       mAP50_95,
        "per_class_AP50": per_class_AP50,
    }


# ─────────────────────────────────────────────────────────────────────────────
# YOLO evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_yolo(
    weights: Path,
    test_images_dir: Path,
    test_labels_dir: Path,
    class_names: List[str],
    device: str = "0",
) -> dict:
    """
    Run YOLOv8 inference on the test set and evaluate with pycocotools.

    conf_thres=0.001 ensures all predictions enter the AP sweep (COCO standard).
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed. Run: pip install ultralytics")

    print(f"  Loading YOLO weights: {weights}")
    model = YOLO(str(weights))

    # Build GT dict first
    print("  Building COCO ground-truth index …")
    coco_gt_dict, fname_to_id = build_coco_gt_from_yolo(
        test_images_dir, test_labels_dir, class_names
    )

    # Run inference
    print("  Running inference on test set …")
    img_paths = sorted(test_images_dir.glob("*.jpg")) + \
                sorted(test_images_dir.glob("*.png"))

    dt_list = []
    for img_path in img_paths:
        results = model.predict(
            source=str(img_path),
            conf=EVAL["conf_thres"],
            iou=EVAL["iou_thres"],
            max_det=EVAL["max_det"],
            device=device,
            verbose=False,
            half=True,   # FP16 matches training
        )
        img_id = fname_to_id.get(img_path.name)
        if img_id is None:
            continue
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            for box, score, cls_id in zip(boxes, scores, cls_ids):
                x1, y1, x2, y2 = box
                dt_list.append({
                    "image_id":    img_id,
                    "category_id": int(cls_id) + 1,    # 1-indexed
                    "bbox":        [float(x1), float(y1),
                                    float(x2 - x1), float(y2 - y1)],
                    "score":       float(score),
                })

    print(f"  Total detections: {len(dt_list)}")
    return run_coco_eval(coco_gt_dict, dt_list, class_names)


# ─────────────────────────────────────────────────────────────────────────────
# Faster R-CNN evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_fasterrcnn(
    weights: Path,
    test_images_dir: Path,
    test_labels_dir: Path,
    class_names: List[str],
    device: str = "cuda",
) -> dict:
    """
    Load Faster R-CNN checkpoint and evaluate on the test set using pycocotools.
    """
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from scripts.train_fasterrcnn import SoybeanDetectionDataset

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # Build model (same architecture as training).
    # weights_backbone=None prevents torchvision from trying to download ImageNet
    # weights on compute nodes that have no internet access.  We load our own
    # fine-tuned weights from the checkpoint below.
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(class_names) + 1)
    ckpt = torch.load(str(weights), map_location=dev)
    # Checkpoint is saved as {"epoch": ..., "model": state_dict, "metrics": ...}
    state_dict = ckpt["model"] if "model" in ckpt else ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(dev)
    model.eval()

    # Dataset / dataloader.
    # SoybeanDetectionDataset.__init__ signature: (data_root, split, transforms=None)
    # test_images_dir == data_root / "images" / split, so we derive both here.
    from torchvision import transforms as T
    _data_root  = test_images_dir.parent.parent   # strip /images/<split>
    _split_name = test_images_dir.name            # e.g. "test"
    dataset = SoybeanDetectionDataset(
        data_root=_data_root,
        split=_split_name,
        transforms=T.ToTensor(),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # Build GT
    print("  Building COCO ground-truth index …")
    coco_gt_dict, _ = build_coco_gt_from_yolo(
        test_images_dir, test_labels_dir, class_names
    )

    print("  Running Faster R-CNN inference on test set …")
    dt_list = []
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(dev) for img in images]
            preds = model(images)
            for pred, tgt in zip(preds, targets):
                img_id = tgt["image_id"].item()
                boxes  = pred["boxes"].cpu().numpy()
                labels = pred["labels"].cpu().numpy()
                scores = pred["scores"].cpu().numpy()
                for box, lbl, score in zip(boxes, labels, scores):
                    if score < EVAL["conf_thres"]:
                        continue
                    x1, y1, x2, y2 = box
                    dt_list.append({
                        "image_id":    int(img_id),
                        "category_id": int(lbl),       # already 1-indexed
                        "bbox":        [float(x1), float(y1),
                                        float(x2 - x1), float(y2 - y1)],
                        "score":       float(score),
                    })

    print(f"  Total detections: {len(dt_list)}")
    return run_coco_eval(coco_gt_dict, dt_list, class_names)


# ─────────────────────────────────────────────────────────────────────────────
# Batch evaluation (--eval_all)
# ─────────────────────────────────────────────────────────────────────────────

# Mapping of experiment name → (model_type, relative weights path)
ALL_EXPERIMENTS = {
    # ── E1: Architectural comparison ─────────────────────────────────────────
    # Three detector paradigms evaluated under identical protocol:
    #   - Two-stage  : Faster R-CNN ResNet50-FPN
    #   - Modern one-stage (2023) : YOLOv8s
    #   - Cutting-edge one-stage (2025) : YOLO26s
    "E1_yolov8s":    ("yolo",       "E1_yolov8s/weights/best.pt"),
    "E1_yolo26s":    ("yolo",       "E1_yolo26s/weights/best.pt"),
    

    # ── E2: Augmentation ablation (YOLOv8s as fixed anchor) ──────────────────
    "E2_no_aug":       ("yolo", "E2_no_aug/weights/best.pt"),
    "E2_yolo_default": ("yolo", "E2_yolo_default/weights/best.pt"),
    "E2_cp_only":      ("yolo", "E2_cp_only/weights/best.pt"),
    "E2_default_cp":   ("yolo", "E2_default_cp/weights/best.pt"),
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained models on the test set")
    p.add_argument("--model_type", choices=["yolo", "fasterrcnn"],
                   help="Model type to evaluate")
    p.add_argument("--weights",    type=str,
                   help="Path to model weights file (best.pt)")
    p.add_argument("--exp_name",   type=str,
                   help="Experiment name; results saved to results/<exp_name>/eval_results.json")
    p.add_argument("--eval_all",   action="store_true",
                   help="Evaluate all experiments defined in ALL_EXPERIMENTS dict")
    p.add_argument("--data_root",  type=str,
                   default=str(PROJECT_ROOT / "soybean_yolo_split"),
                   help="Root directory of the dataset split")
    p.add_argument("--split",      type=str, default="test",
                   choices=["val", "test"],
                   help="Dataset split to evaluate on (default: test)")
    p.add_argument("--device",     type=str, default="0",
                   help="Device: '0' for GPU 0, 'cpu' for CPU")
    return p.parse_args()


def evaluate_one(
    exp_name: str,
    model_type: str,
    weights: Path,
    data_root: Path,
    split: str,
    device: str,
    class_names: List[str],
) -> dict:
    print(f"\n{'─'*60}")
    print(f"  Evaluating: {exp_name}  [{model_type}]  split={split}")
    print(f"  Weights:    {weights}")
    print(f"{'─'*60}")

    test_images = data_root / "images" / split
    test_labels = data_root / "labels" / split

    t0 = time.time()
    if model_type == "yolo":
        metrics = evaluate_yolo(weights, test_images, test_labels, class_names, device)
    else:
        dev = f"cuda:{device}" if device.isdigit() else device
        metrics = evaluate_fasterrcnn(weights, test_images, test_labels, class_names, dev)
    elapsed = time.time() - t0

    metrics["exp_name"]    = exp_name
    metrics["model_type"]  = model_type
    metrics["split"]       = split
    metrics["eval_time_s"] = elapsed
    metrics["weights"]     = str(weights)

    # Save per-experiment results
    out_dir = RESULTS_ROOT / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  mAP@0.50      = {metrics['mAP50']:.4f}")
    print(f"  mAP@0.50:0.95 = {metrics['mAP50_95']:.4f}")
    print(f"  Per-class AP@0.50:")
    for cls, ap in metrics["per_class_AP50"].items():
        print(f"    {cls:30s}: {ap:.4f}")
    print(f"  Saved → {out_path}")

    return metrics


def main() -> None:
    args = parse_args()
    class_names = DATASET["names"]
    data_root   = Path(args.data_root)

    all_results = {}

    if args.eval_all:
        for exp_name, (model_type, rel_weights) in ALL_EXPERIMENTS.items():
            weights = RESULTS_ROOT / rel_weights
            if not weights.exists():
                print(f"  [SKIP] {exp_name}: weights not found at {weights}")
                continue
            m = evaluate_one(exp_name, model_type, weights, data_root,
                             args.split, args.device, class_names)
            all_results[exp_name] = m
    else:
        if not args.model_type or not args.weights or not args.exp_name:
            sys.exit("Provide --model_type, --weights, --exp_name  OR  use --eval_all")
        weights = Path(args.weights)
        m = evaluate_one(args.exp_name, args.model_type, weights, data_root,
                         args.split, args.device, class_names)
        all_results[args.exp_name] = m

    # Save consolidated summary
    summary_path = RESULTS_ROOT / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nConsolidated results → {summary_path}")


if __name__ == "__main__":
    main()
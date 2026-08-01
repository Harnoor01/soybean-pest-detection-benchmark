#!/usr/bin/env python3
"""
benchmark_speed.py
==================
Measures single-image inference latency (ms/image) and throughput (FPS)
for all trained models.

Methodology
-----------
  1. Warm up: run N_WARMUP forward passes on a blank tensor (GPU spins up,
     JIT caches compiled, memory allocated) — these are discarded.
  2. Timed: run N_RUNS forward passes with torch.cuda.synchronize() before
     each timer start and stop to accurately measure GPU wall-time.
  3. Report: mean ± std latency (ms) and FPS over N_RUNS.

This matches the benchmarking protocol used in RT-DETR, YOLOv8, and
Faster R-CNN speed comparisons (single image, GPU, FP16 for YOLO / FP32
for Faster RCNN to reflect actual deployment configuration).

Usage
-----
  python benchmark_speed.py --exp_name E1_yolov8s \
      --model_type yolo \
      --weights results/E1_yolov8s/weights/best.pt

  # Benchmark all at once:
  python benchmark_speed.py --benchmark_all

Output
------
  results/<exp_name>/speed_results.json
  results/speed_summary.json            (consolidated)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.experiment_config import BENCHMARK, DATASET, RESULTS_ROOT


# ─────────────────────────────────────────────────────────────────────────────
# Constants — match BENCHMARK config
# ─────────────────────────────────────────────────────────────────────────────
N_WARMUP = BENCHMARK["n_warmup"]   # 50
N_RUNS   = BENCHMARK["n_runs"]     # 200
IMG_SIZE = 640                      # YOLO input resolution
DEVICE   = BENCHMARK["device"]     # "cuda"


def _sync():
    """Synchronise CUDA stream so timer reflects actual GPU completion."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_dummy_input(height: int = 640, width: int = 640, device: torch.device = None) -> torch.Tensor:
    """Random float32 image tensor in [0, 1] with shape (1, 3, H, W)."""
    t = torch.rand(1, 3, height, width, dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t


# ─────────────────────────────────────────────────────────────────────────────
# YOLO benchmarking
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_yolo(weights: Path, device_str: str = "0") -> Dict[str, float]:
    """
    Benchmark a YOLOv8 model using ultralytics' built-in predict API.

    FP16 (half=True) is used, matching the AMP training setting.
    Input: 640 × 640 pixel single image.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed.")

    model = YOLO(str(weights))
    # Create a real 640×640 numpy image (uint8) so ultralytics processes it
    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Warm-up
    print(f"    Warm-up ({N_WARMUP} passes) …")
    for _ in range(N_WARMUP):
        model.predict(dummy_img, device=device_str, verbose=False,
                      half=BENCHMARK["half"], imgsz=IMG_SIZE)

    # Timed runs
    print(f"    Timing ({N_RUNS} passes) …")
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        model.predict(dummy_img, device=device_str, verbose=False,
                      half=BENCHMARK["half"], imgsz=IMG_SIZE)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)   # ms

    return _summarise(times)


# ─────────────────────────────────────────────────────────────────────────────
# Faster R-CNN benchmarking
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_fasterrcnn(weights: Path, device_str: str = "cuda") -> Dict[str, float]:
    """
    Benchmark Faster R-CNN using a raw PyTorch forward pass.

    FP32 (float) is used for Faster R-CNN (matching deployment).
    Input: 800 × 1333 pixel tensor (standard multi-scale inference size).

    Note: Faster R-CNN's variable-size multi-scale resizing means the
    inference time depends on input resolution.  We use 800×1333 as this
    is the default during training.
    """
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    n_classes = len(DATASET["names"])
    dev = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # weights_backbone=None prevents torchvision downloading ImageNet weights
    # on compute nodes with no internet.  We load our checkpoint state dict below.
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes + 1)

    ckpt = torch.load(str(weights), map_location=dev)
    # Checkpoint is saved as {"epoch": ..., "model": state_dict, "metrics": ...}
    state_dict = ckpt["model"] if "model" in ckpt else ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(dev).eval()

    dummy = [make_dummy_input(800, 1333, dev).squeeze(0)]   # list of (3, H, W)

    # Warm-up
    print(f"    Warm-up ({N_WARMUP} passes) …")
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _sync()
            model(dummy)
            _sync()

    # Timed runs
    print(f"    Timing ({N_RUNS} passes) …")
    times = []
    with torch.no_grad():
        for _ in range(N_RUNS):
            _sync()
            t0 = time.perf_counter()
            model(dummy)
            _sync()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)   # ms

    return _summarise(times)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _summarise(times_ms: list) -> Dict[str, float]:
    arr = np.array(times_ms)
    mean = float(arr.mean())
    std  = float(arr.std())
    p50  = float(np.percentile(arr, 50))
    p95  = float(np.percentile(arr, 95))
    fps  = 1000.0 / mean
    return {
        "mean_ms":  round(mean, 3),
        "std_ms":   round(std,  3),
        "p50_ms":   round(p50,  3),
        "p95_ms":   round(p95,  3),
        "fps":      round(fps,  2),
        "n_runs":   len(times_ms),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch config
# ─────────────────────────────────────────────────────────────────────────────

ALL_BENCHMARK_EXPERIMENTS = {
    # E1: architectural comparison — benchmark all three paradigms
    "E1_yolov8s":    ("yolo",       "E1_yolov8s/weights/best.pt"),
    "E1_yolo26s":    ("yolo",       "E1_yolo26s/weights/best.pt"),
    
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark inference speed of trained models")
    p.add_argument("--exp_name",       type=str, help="Experiment name")
    p.add_argument("--model_type",     choices=["yolo", "fasterrcnn"],
                   help="Model type")
    p.add_argument("--weights",        type=str, help="Path to weights file")
    p.add_argument("--benchmark_all",  action="store_true",
                   help="Benchmark all experiments in ALL_BENCHMARK_EXPERIMENTS")
    p.add_argument("--device",         type=str, default="0",
                   help="Device: '0' for GPU 0, 'cpu' for CPU")
    return p.parse_args()


def benchmark_one(exp_name: str, model_type: str, weights: Path,
                  device: str) -> dict:
    print(f"\n{'─'*60}")
    print(f"  Benchmarking: {exp_name}  [{model_type}]")
    print(f"  Weights:      {weights}")
    print(f"  Device:       {device}  |  n_warmup={N_WARMUP}  n_runs={N_RUNS}")
    print(f"{'─'*60}")

    if model_type == "yolo":
        results = benchmark_yolo(weights, device)
    else:
        dev_str = f"cuda:{device}" if device.isdigit() else device
        results = benchmark_fasterrcnn(weights, dev_str)

    results["exp_name"]   = exp_name
    results["model_type"] = model_type

    print(f"\n  Latency : {results['mean_ms']:.1f} ± {results['std_ms']:.1f} ms")
    print(f"  P50     : {results['p50_ms']:.1f} ms")
    print(f"  P95     : {results['p95_ms']:.1f} ms")
    print(f"  FPS     : {results['fps']:.1f}")

    out_dir = RESULTS_ROOT / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "speed_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {out_path}")

    return results


def main() -> None:
    args = parse_args()
    all_results = {}

    if args.benchmark_all:
        for exp_name, (model_type, rel_w) in ALL_BENCHMARK_EXPERIMENTS.items():
            weights = RESULTS_ROOT / rel_w
            if not weights.exists():
                print(f"  [SKIP] {exp_name}: weights not found at {weights}")
                continue
            r = benchmark_one(exp_name, model_type, weights, args.device)
            all_results[exp_name] = r
    else:
        if not (args.exp_name and args.model_type and args.weights):
            sys.exit("Provide --exp_name, --model_type, --weights  OR  --benchmark_all")
        weights = Path(args.weights)
        r = benchmark_one(args.exp_name, args.model_type, weights, args.device)
        all_results[args.exp_name] = r

    summary_path = RESULTS_ROOT / "speed_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSpeed summary → {summary_path}")


if __name__ == "__main__":
    main()
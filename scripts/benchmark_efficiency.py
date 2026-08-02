#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import statistics

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


DEVICE = "cuda:0"
IMAGE_SIZE = 640
BATCH_SIZE = 1
WARMUP_RUNS = 50
TIMED_RUNS = 300
USE_FP16 = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark model parameters, GFLOPs, latency, and FPS."
    )
    parser.add_argument(
        "--yolov8s",
        required=True,
        help="Path to the trained YOLOv8s checkpoint.",
    )
    parser.add_argument(
        "--yolo26s",
        required=True,
        help="Path to the trained YOLO26s checkpoint.",
    )
    parser.add_argument(
        "--rtdetr",
        required=True,
        help="Path to the trained RT-DETR-L checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="efficiency_results.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def get_model_complexity(checkpoint: str) -> tuple[int, float]:
    wrapper = YOLO(checkpoint)
    model = wrapper.model

    parameters = sum(parameter.numel() for parameter in model.parameters())
    gflops = get_flops(model, imgsz=IMAGE_SIZE)

    return parameters, gflops


def benchmark_latency(checkpoint: str) -> tuple[float, float]:
    wrapper = YOLO(checkpoint)
    model = wrapper.model.to(DEVICE)
    model.eval()

    if USE_FP16:
        model.half()

    dtype = torch.float16 if USE_FP16 else torch.float32

    dummy_input = torch.randn(
        BATCH_SIZE,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=DEVICE,
        dtype=dtype,
    )

    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            model(dummy_input)

    torch.cuda.synchronize()

    timings_ms = []

    with torch.inference_mode():
        for _ in range(TIMED_RUNS):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            model(dummy_input)
            end_event.record()

            torch.cuda.synchronize()
            timings_ms.append(start_event.elapsed_time(end_event))

    mean_latency_ms = statistics.mean(timings_ms)
    fps = 1000.0 * BATCH_SIZE / mean_latency_ms

    return mean_latency_ms, fps


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    checkpoints = {
        "YOLOv8s": args.yolov8s,
        "YOLO26s": args.yolo26s,
        "RT-DETR-L": args.rtdetr,
    }

    print("GPU:", torch.cuda.get_device_name(0))
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Precision:", "FP16" if USE_FP16 else "FP32")

    rows = []

    for model_name, checkpoint in checkpoints.items():
        if not Path(checkpoint).exists():
            raise FileNotFoundError(
                f"Checkpoint does not exist: {checkpoint}"
            )

        print(f"\nBenchmarking {model_name}")

        parameters, gflops = get_model_complexity(checkpoint)
        latency_ms, fps = benchmark_latency(checkpoint)

        print(f"Latency: {latency_ms:.3f} ms/image")
        print(f"FPS: {fps:.2f}")

        rows.append(
            {
                "model": model_name,
                "parameters": parameters,
                "parameters_millions": parameters / 1_000_000,
                "gflops": gflops,
                "latency_mean_ms": latency_ms,
                "latency_sd_ms": 0.0,
                "fps_mean": fps,
                "fps_sd": 0.0,
            }
        )

    output_file = Path(args.output)

    with output_file.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to: {output_file}")

    for row in rows:
        print(
            f"{row['model']}: "
            f"{row['parameters_millions']:.2f} M parameters, "
            f"{row['gflops']:.2f} GFLOPs, "
            f"{row['latency_mean_ms']:.2f} ms, "
            f"{row['fps_mean']:.2f} FPS"
        )


if __name__ == "__main__":
    main()

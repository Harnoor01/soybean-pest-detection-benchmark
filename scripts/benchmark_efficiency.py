from pathlib import Path
import csv
import statistics

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


CHECKPOINTS = {
    "YOLOv8s": {
        42: (
            "/project/def-grandha8-ab/harnoor1/pest_project_EPA/"
            "soybean_detection_project/results/E1_yolov8s/weights/best.pt"
        ),
    },
    "YOLO26s": {
        42: (
            "/project/def-grandha8-ab/harnoor1/pest_project_EPA/"
            "soybean_detection_project/results/E1_yolo26s/weights/best.pt"
        ),
    },
    "RT-DETR-L": {
        42: (
            "/project/def-grandha8-ab/harnoor1/pest_project_EPA/"
            "soybean_detection_project/results/E1_rtdetr/weights/best.pt"
        ),
    },
}


DEVICE = "cuda:0"
IMAGE_SIZE = 640
BATCH_SIZE = 1
WARMUP_RUNS = 50
TIMED_RUNS = 300
USE_FP16 = True


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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Precision:", "FP16" if USE_FP16 else "FP32")

    rows = []

    for model_name, seeds in CHECKPOINTS.items():
        latency_values = []
        fps_values = []

        first_checkpoint = next(iter(seeds.values()))

        if not Path(first_checkpoint).exists():
            raise FileNotFoundError(
                f"Checkpoint does not exist: {first_checkpoint}"
            )

        parameters, gflops = get_model_complexity(first_checkpoint)

        for seed, checkpoint in seeds.items():
            if not Path(checkpoint).exists():
                raise FileNotFoundError(
                    f"Checkpoint does not exist: {checkpoint}"
                )

            print(f"\nBenchmarking {model_name}, seed {seed}")

            latency_ms, fps = benchmark_latency(checkpoint)

            print(f"Latency: {latency_ms:.3f} ms/image")
            print(f"FPS: {fps:.2f}")

            latency_values.append(latency_ms)
            fps_values.append(fps)

        row = {
            "model": model_name,
            "parameters": parameters,
            "parameters_millions": parameters / 1_000_000,
            "gflops": gflops,
            "latency_mean_ms": statistics.mean(latency_values),
            "latency_sd_ms": (
                statistics.stdev(latency_values)
                if len(latency_values) > 1
                else 0.0
            ),
            "fps_mean": statistics.mean(fps_values),
            "fps_sd": (
                statistics.stdev(fps_values)
                if len(fps_values) > 1
                else 0.0
            ),
        }

        rows.append(row)

    output_file = Path(
        "/project/def-grandha8-ab/harnoor1/pest_project_EPA/"
        "soybean_detection_project/efficiency_results.csv"
    )

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
            f"{row['latency_mean_ms']:.2f} ± "
            f"{row['latency_sd_ms']:.2f} ms, "
            f"{row['fps_mean']:.2f} ± "
            f"{row['fps_sd']:.2f} FPS"
        )


if __name__ == "__main__":
    main()
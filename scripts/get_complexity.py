#!/usr/bin/env python3

import argparse

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate parameter counts and GFLOPs for trained models."
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
        "--imgsz",
        type=int,
        default=640,
        help="Input image size used for GFLOPs calculation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoints = {
        "YOLOv8s": args.yolov8s,
        "YOLO26s": args.yolo26s,
        "RT-DETR-L": args.rtdetr,
    }

    for model_name, checkpoint in checkpoints.items():
        wrapper = YOLO(checkpoint)
        model = wrapper.model

        parameters = sum(parameter.numel() for parameter in model.parameters())
        gflops = get_flops(model, imgsz=args.imgsz)

        print(
            f"{model_name}: "
            f"{parameters / 1_000_000:.2f} M parameters, "
            f"{gflops:.2f} GFLOPs"
        )


if __name__ == "__main__":
    main()

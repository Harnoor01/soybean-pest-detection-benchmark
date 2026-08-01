from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


CHECKPOINTS = {
    "YOLOv8s": (
        "/project/def-grandha8-ab/harnoor1/pest_project_EPA/"
        "soybean_detection_project/results/E1_yolov8s/weights/best.pt"
    ),
    "YOLO26s": (
        "/project/def-grandha8-ab/harnoor1/pest_project_EPA/"
        "soybean_detection_project/results/E1_yolo26s/weights/best.pt"
    ),
    "RT-DETR-L": (
        "/project/def-grandha8-ab/harnoor1/pest_project_EPA/"
        "soybean_detection_project/results/E1_rtdetr/weights/best.pt"
    ),
}


for model_name, checkpoint in CHECKPOINTS.items():
    wrapper = YOLO(checkpoint)
    model = wrapper.model

    parameters = sum(p.numel() for p in model.parameters())
    gflops = get_flops(model, imgsz=640)

    print(
        f"{model_name}: "
        f"{parameters / 1_000_000:.2f} M parameters, "
        f"{gflops:.2f} GFLOPs"
    )
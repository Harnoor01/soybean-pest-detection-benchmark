"""
experiment_config.py
====================
Central configuration for all soybean pest detection experiments.

WHY A CENTRAL CONFIG:
  - Guarantees identical hyperparameters across all models → fair comparison.
  - Single source of truth; changing one value propagates everywhere.
  - Forces explicit documentation of every hyperparameter choice.

MODELS COVERED (E1 — Architectural Comparison):
  - YOLOv8s    : CNN one-stage, anchor-free, DFL, NMS           (Ultralytics 2023)
  - RT-DETR-L  : Transformer-based, attention encoder, NMS-free (Baidu/Ultralytics 2023)
  - YOLO26s    : CNN one-stage, NMS-free, STAL, MuSGD           (Ultralytics 2025)

EXPERIMENT STRUCTURE:
  E1 — Architecture comparison : all three models, yolo_default augmentation
  E2 — Augmentation ablation   : YOLOv8s only, four augmentation conditions
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# PATHS  (edit DATA_ROOT to match your Narval scratch path)
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # soybean_detection_project/
DATA_ROOT    = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "soybean_yolo_split"))
RESULTS_ROOT = PROJECT_ROOT / "results"
LOGS_ROOT    = PROJECT_ROOT / "logs"

# ──────────────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────────────
DATASET = {
    "yaml":       str(PROJECT_ROOT / "data" / "data.yaml"),
    "nc":         4,
    "names":      ["Eocanthecona_Bug", "Tobacco_Caterpillar",
                   "Red_Hairy_Caterpillar", "Larva_Spodoptera"],
    "n_train":    2664,   # 70 % of 3807
    "n_val":      761,    # 20 %
    "n_test":     382,    # 10 %
}

# ──────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42   # fixed across all experiments; reported in paper §Experiments

# ──────────────────────────────────────────────────────────────────────────────
# SHARED TRAINING SETTINGS  (identical across all models → fair comparison)
# ──────────────────────────────────────────────────────────────────────────────
# WHY imgsz=640:
#   Dataset analysis shows mean bbox width ≈ 173 px and mean bbox height ≈ 140 px
#   at 640-px YOLO input space.  All objects are "medium" (32–96 px) to "large"
#   (>96 px) by COCO definition.  640 px is YOLO's standard input resolution and
#   provides sufficient detail without excessive memory overhead on A100 40 GB.
#
# WHY epochs=100:
#   2664 train images / batch_16 ≈ 167 iterations/epoch → 16 700 total steps.
#   Comparable to fine-tuning schedules on similarly sized agricultural datasets
#   (e.g., PlantDoc, IP102 subsets).  patience=50 gives the scheduler room to
#   converge without the full 100 epochs when the dataset is easy.
#
# WHY batch=16:
#   Peak VRAM for YOLOv8m at imgsz=640 batch=16 is ~18 GB on a single A100 40 GB
#   (measured empirically); safe for all three variants (n/s/m).  Using the same
#   batch across variants keeps gradient-noise scale constant → fair comparison.
#
# WHY lr0=0.01 + cosine decay to lrf=0.01:
#   Standard YOLO learning-rate schedule proven across hundreds of datasets.
#   Final LR = lr0 × lrf = 1e-4, giving stable fine-tuning plateau.
#
# WHY optimizer=SGD, momentum=0.937, weight_decay=5e-4:
#   SGD with Nesterov momentum generalises better than Adam for detection
#   fine-tuning (Goyal et al. 2017 "Accurate Large Minibatch SGD").
#   weight_decay=5e-4 is the canonical value from Faster R-CNN, SSD, YOLO papers.
#
# WHY warmup_epochs=3:
#   With COCO pre-trained weights, the head layers are randomly initialised for 4
#   classes; warmup prevents large early gradient updates from corrupting the
#   backbone features before the head converges.
#
# WHY patience=50:
#   50 epochs without mAP improvement is a generous stopping criterion that
#   avoids premature termination while still preventing overfitting on a ~2 k
#   training set.
#
# WHY workers=8:
#   Narval A100 nodes have ≥ 32 CPU cores; 8 DataLoader workers saturates the
#   GPU without excessive process overhead.

TRAIN_SHARED = {
    "imgsz":          640,
    "epochs":         100,
    "batch":          16,
    "lr0":            0.01,
    "lrf":            0.01,      # final LR = lr0 * lrf  (cosine schedule)
    "momentum":       0.937,
    "weight_decay":   0.0005,
    "warmup_epochs":  3,
    "warmup_momentum":0.8,
    "warmup_bias_lr": 0.1,
    "optimizer":      "SGD",     # explicit SGD for YOLOv8s — do NOT use for YOLO26s
    "patience":       50,
    "workers":        8,
    "seed":           SEED,
    "deterministic":  True,
    "exist_ok":       False,
    "pretrained":     True,      # always fine-tune from COCO weights
    "device":         0,         # GPU 0; set "cpu" for CPU-only
    "amp":            True,      # mixed-precision (FP16) → 2× throughput on A100
    "verbose":        True,
    "save":           True,
    "save_period":    -1,        # only save best + last checkpoints
    "val":            True,
    "plots":          True,
    "close_mosaic":   10,        # disable mosaic in final 10 epochs
}

# ──────────────────────────────────────────────────────────────────────────────
# YOLO26-SPECIFIC TRAINING SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
# WHY optimizer="auto" for YOLO26:
#   YOLO26 ships with the MuSGD optimizer (SGD + Muon hybrid), which is part of
#   its architectural identity and contributes to its reported performance gains.
#   Overriding to plain SGD would invalidate the architecture comparison — we
#   would no longer be evaluating YOLO26 as designed.  optimizer="auto" tells
#   Ultralytics to use the model's recommended default (MuSGD for YOLO26).
#
#   All other parameters (imgsz, epochs, batch, lr0, warmup, patience) are
#   identical to TRAIN_SHARED so that the only variable between YOLOv8s and
#   YOLO26s is the model architecture itself.  This is the correct fair-comparison
#   design for an architectural benchmark paper.
#
# WHY same lr0=0.01:
#   MuSGD operates on the same learning rate schedule as SGD; the Ultralytics
#   framework applies the cosine decay identically.  Using the same lr0 ensures
#   the learning rate trajectory is comparable between experiments.

YOLO26_TRAIN_SHARED = {
    **TRAIN_SHARED,             # inherit all shared params
    "optimizer": "auto",        # use YOLO26's native MuSGD (overrides SGD above)
}

# ──────────────────────────────────────────────────────────────────────────────
# RT-DETR-L TRAINING SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
# WHY optimizer="AdamW" for RT-DETR:
#   Transformer architectures are universally trained with Adam-family optimizers.
#   SGD does not converge stably with transformer attention layers.  AdamW is the
#   standard choice in DETR, RT-DETR, and all transformer detection papers.
#
# WHY lr0=0.0001:
#   RT-DETR's published training uses lr=0.0001 (Zhao et al. 2023).  This is 100×
#   lower than YOLO's lr0=0.01.  Transformers are sensitive to high LR — using
#   YOLO's lr would diverge immediately.  This is NOT a controlled variable;
#   it is the correct published lr for this architecture.
#
# WHY warmup_epochs=0:
#   RT-DETR initialises from COCO-pretrained weights with the full transformer
#   already trained.  The head is adapted during fine-tuning with AdamW's natural
#   adaptive step control, making a separate warmup phase unnecessary.
#
# WHY same imgsz=640, epochs=100, batch=16:
#   We hold these identical to YOLO experiments.  RT-DETR-L is tested at 640px in
#   its original paper.  100 epochs with patience=50 is sufficient given our
#   dataset size.  batch=16 fits A100 40GB for RT-DETR-L at 640px.
#
# NOTE on augmentation:
#   In E1, RT-DETR receives the same yolo_default augmentation flags as YOLO
#   models.  Ultralytics applies the compatible subset (flips, HSV, scale) and
#   silently ignores YOLO-specific ops (mosaic, copy_paste) for DETR models.
#   This is the correct behaviour — we are not disadvantaging RT-DETR.

RTDETR_TRAIN_SHARED = {
    "imgsz":         640,
    "epochs":        100,
    "batch":         16,
    "lr0":           0.0001,    # RT-DETR published learning rate
    "lrf":           0.01,      # final LR = lr0 × lrf = 1e-6 (cosine decay floor)
    "weight_decay":  0.0001,    # AdamW standard weight decay for transformers
    "optimizer":     "AdamW",   # required for transformer convergence
    "warmup_epochs": 0,         # not needed with COCO pretrained transformer
    "patience":      50,
    "workers":       8,
    "seed":          SEED,
    "deterministic": True,
    "exist_ok":      False,
    "pretrained":    True,
    "device":        0,
    "amp":           True,
    "verbose":       True,
    "save":          True,
    "save_period":   -1,
    "val":           True,
    "plots":         True,
    "close_mosaic":  0,         # RT-DETR does not use mosaic
}

# ──────────────────────────────────────────────────────────────────────────────
# AUGMENTATION CONFIGURATIONS  (E3: ablation study)
# ──────────────────────────────────────────────────────────────────────────────
# WHY flipud=0 in all conditions:
#   Agricultural field images are taken from above/side; vertically flipped
#   insects are physically unrealistic and may confuse the model.
#
# WHY fliplr=0.5:
#   Left–right orientation is arbitrary → symmetric augmentation appropriate.
#
# WHY hsv_s=0.7, hsv_v=0.4:
#   High saturation and value jitter accounts for the wide range of field
#   lighting conditions (direct sun, shade, overcast) in the dataset.
#
# WHY scale=0.5 in YOLO defaults:
#   Simulates variation in camera distance.  Our analysis shows bbox sizes vary
#   from ~100 px to >400 px in YOLO space even within one class, so scale
#   augmentation is well-motivated.
#
# WHY copy_paste=0.5 in CP conditions:
#   With ~380 unique specimens/class, pasting instances across images synthesises
#   novel specimen-background combinations.  p=0.5 is the standard value from
#   Ghiasi et al. (CVPR 2021).  YOLO's copy_paste requires mosaic=1.0 to operate.

AUG_NO_AUG = dict(
    # All geometric and colour augmentations disabled.
    # Used as lower-bound baseline to quantify total augmentation benefit.
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    degrees=0.0, translate=0.0, scale=0.0,
    shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.0,
    mosaic=0.0, mixup=0.0, copy_paste=0.0,
    erasing=0.0,
)

AUG_YOLO_DEFAULT = dict(
    # Standard YOLOv8 augmentation pipeline (ultralytics defaults).
    # Serves as the primary comparison point against the copy-paste condition.
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,     # see justification above
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    erasing=0.0,
)

AUG_COPY_PASTE_ONLY = dict(
    # Copy-paste enabled, all other geometric transforms disabled.
    # Isolates the contribution of instance pasting.
    # mosaic=1.0 is required for YOLO's copy_paste to activate.
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,   # minimal colour aug kept
    degrees=0.0, translate=0.0, scale=0.0,
    shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.5,
    erasing=0.0,
)

AUG_YOLO_DEFAULT_PLUS_CP = dict(
    # Full YOLO augmentation pipeline + copy-paste.
    # Expected to be the strongest condition.
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=0.0, translate=0.1, scale=0.5,
    shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.5,
    erasing=0.0,
)

# ── Corrected bbox copy-paste conditions (E3 experiments) ────────────────────
# These conditions are used with the OFFLINE copy-paste augmented dataset
# generated by scripts/generate_cp_dataset.py.
# Ultralytics copy_paste=0 in all E3 conditions because copy-paste was already
# applied offline (Ultralytics copy_paste requires segmentation masks, which
# InsectBase does not provide — confirmed in v8.4.56 source).

AUG_BBOX_CP_ONLY = dict(
    # Offline bbox copy-paste applied to training images.
    # YOLO augmentation: HSV + flip only (no mosaic, no geometric distortion).
    # This isolates the effect of copy-paste from mosaic/scale/translate.
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=0.0, translate=0.0, scale=0.0,
    shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.5,
    mosaic=0.0,   # no mosaic — isolates copy-paste contribution
    mixup=0.0, copy_paste=0.0, erasing=0.0,
)

AUG_YOLO_DEFAULT_BBOX_CP = dict(
    # Offline bbox copy-paste applied to training images,
    # PLUS standard YOLO mosaic augmentation on top.
    # Tests whether copy-paste stacks additively with YOLO augmentation.
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=0.0, translate=0.1, scale=0.5,
    shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.5,
    mosaic=1.0,
    mixup=0.0, copy_paste=0.0, erasing=0.0,
)

# ── E2 Close-Mosaic Scheduling Conditions ────────────────────────────────────
# WHY study close_mosaic:
#   Mosaic forces pests to appear at quarter-canvas size (smaller than real).
#   This helps detection rate but hurts bounding-box precision because the model
#   never sees pests at their natural full scale during late training.
#   close_mosaic=N disables mosaic for the final N epochs, allowing the model
#   to fine-tune on full-scale natural images — improving mAP@0.5:0.95.
#   InsectBase pests appear at consistent scale in field photography, so a
#   longer full-scale fine-tuning phase is expected to help localisation.
#
# All three conditions share identical augmentation (YOLO default):
#   HSV jitter, horizontal flip, mosaic=1.0, scale=0.5, translate=0.1.
#   The ONLY variable is when mosaic switches off.
#
# close_mosaic overrides TRAIN_SHARED["close_mosaic"]=10 when merged in train_yolo.py.

AUG_CLOSE_MOSAIC_0 = dict(
    # Mosaic active for ALL 100 epochs — never disabled.
    # Lower bound for localisation precision: model never sees full-scale images.
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=0.0, translate=0.1, scale=0.5,
    shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.5,
    mosaic=1.0, mixup=0.0, copy_paste=0.0, erasing=0.0,
    close_mosaic=0,    # never disable mosaic
)

AUG_CLOSE_MOSAIC_10 = dict(
    # Mosaic disabled for final 10 epochs (YOLO default).
    # 10 full-scale fine-tuning epochs out of 100.
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=0.0, translate=0.1, scale=0.5,
    shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.5,
    mosaic=1.0, mixup=0.0, copy_paste=0.0, erasing=0.0,
    close_mosaic=10,   # YOLO default — reference condition
)

AUG_CLOSE_MOSAIC_50 = dict(
    # Mosaic disabled for final 50 epochs.
    # 50 full-scale fine-tuning epochs — hypothesis: better mAP@0.5:0.95
    # because InsectBase pests appear at consistent field-photography scale.
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=0.0, translate=0.1, scale=0.5,
    shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.5,
    mosaic=1.0, mixup=0.0, copy_paste=0.0, erasing=0.0,
    close_mosaic=50,   # extended full-scale phase
)

ABLATION_CONDITIONS = {
    # E2 — geometric augmentation study (3 conditions reported in paper):
    "no_aug":            AUG_NO_AUG,
    "yolo_default":      AUG_YOLO_DEFAULT,       # close_mosaic=10 from TRAIN_SHARED
    "copy_paste_only":   AUG_COPY_PASTE_ONLY,    # mosaic-only (no geometric distortion)
    "yolo_default_cp":   AUG_YOLO_DEFAULT_PLUS_CP,
    # E2 new — close_mosaic scheduling ablation (3 conditions):
    "close_mosaic_0":    AUG_CLOSE_MOSAIC_0,
    "close_mosaic_10":   AUG_CLOSE_MOSAIC_10,
    "close_mosaic_50":   AUG_CLOSE_MOSAIC_50,
    # E3 conditions (offline bbox copy-paste augmented dataset):
    "bbox_cp_only":         AUG_BBOX_CP_ONLY,
    "yolo_default_bbox_cp": AUG_YOLO_DEFAULT_BBOX_CP,
}

# ──────────────────────────────────────────────────────────────────────────────
# FASTER R-CNN SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
# WHY imgsz_min=800, imgsz_max=1333:
#   Faster R-CNN uses multi-scale training; 800/1333 is the standard ResNet50-FPN
#   configuration from Lin et al. (FPN, CVPR 2017) and Ren et al. (Faster RCNN).
#
# WHY batch=4, accum_steps=4 → effective batch=16:
#   Faster R-CNN's RPN + RoI heads are memory-intensive; batch=4 is the limit on
#   A100 40 GB at 800-px input.  Gradient accumulation over 4 steps matches the
#   effective batch of YOLOv8 experiments → comparable gradient statistics.
#
# WHY lr=0.005:
#   Half of the canonical COCO Faster R-CNN LR (0.02) scaled for our smaller
#   dataset (2664 vs. 117 k COCO images).  Using the linear scaling rule
#   (Goyal 2017): lr ∝ batch × dataset_scale → 0.02 × (16/32) × (2664/117266)
#   ≈ 0.0023; we round up to 0.005 for stability.
#
# WHY epochs=26 with milestones [16, 22]:
#   Equivalent to the standard 1× COCO training schedule scaled to our dataset.
#   LR decays by 0.1 at epochs 16 and 22 (MultiStepLR).

FASTERRCNN = {
    "backbone":          "resnet50",
    "pretrained_backbone": True,
    "pretrained_coco":   True,       # initialise from COCO detection weights
    "imgsz_min":         800,
    "imgsz_max":         1333,
    "batch":             4,
    "accum_steps":       4,          # gradient accumulation → effective batch 16
    "epochs":            26,
    "lr":                0.005,
    "momentum":          0.9,
    "weight_decay":      0.0005,
    "lr_milestones":     [16, 22],
    "lr_gamma":          0.1,
    "warmup_iters":      500,        # linear warmup for 500 iterations
    "num_workers":       8,
    "amp":               True,
    "seed":              SEED,
    # RPN / RoI head settings (keep torchvision defaults for fair baseline)
    "rpn_nms_thresh":    0.7,
    "box_score_thresh":  0.05,
    "box_nms_thresh":    0.5,
    "box_detections_per_img": 100,
}

# ──────────────────────────────────────────────────────────────────────────────
# SPEED BENCHMARKING
# ──────────────────────────────────────────────────────────────────────────────
BENCHMARK = {
    "n_warmup":   50,    # discard first 50 inferences (GPU warm-up)
    "n_runs":     200,   # average over 200 inferences
    "batch":      1,     # single-image latency (deployment scenario)
    "half":       True,  # FP16 for YOLO (matches training); FP32 for Faster RCNN
    "device":     "cuda",
}

# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION THRESHOLDS
# ──────────────────────────────────────────────────────────────────────────────
EVAL = {
    "conf_thres":    0.001,   # low conf for mAP sweep (do NOT use 0.25 for eval)
    "iou_thres":     0.6,     # NMS IoU threshold
    "max_det":       300,
    "iou_map_thres": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
}
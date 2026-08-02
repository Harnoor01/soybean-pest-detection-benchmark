#!/usr/bin/env python3
"""
split_yolo_specimen.py
----------------------
Specimen-level dataset splitter for InsectBase.

PROBLEM WITH IMAGE-LEVEL SPLIT:
  Each physical insect was photographed ~6-10 times in a burst session.
  A random image-level shuffle scatters these burst images across train/val/test,
  so the model sees the same individual in training and is tested on different
  angles of the same individual. This inflates mAP scores.

THIS SCRIPT:
  Groups all burst images of the same physical specimen together, then
  splits at the SPECIMEN level (70/20/10). This guarantees that every
  individual insect appears in exactly one split.

Filename format:
  ClassName__NNNNNIMGmmmm_NNNNN_BURST{datetime}.jpg
  Specimen ID = ClassName + "__" + NNNNN  (e.g. "Eocanthecona_Bug_A__00001")

Usage (run from soybean_detection_project/):
    python scripts/split_yolo_specimen.py --seed 42
    python scripts/split_yolo_specimen.py --seed 0
    python scripts/split_yolo_specimen.py --seed 123

Output:
    soybean_yolo_splits/seed{N}_specimen/
        images/train/  images/val/  images/test/
        labels/train/  labels/val/  labels/test/
        data.yaml
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
SRC_DEFAULT = PROJECT / "soybean_yolo_clean"
DST_ROOT    = PROJECT / "soybean_yolo_splits"

CLASS_NAMES = {
    0: "Eocanthecona_Bug",
    1: "Tobacco_Caterpillar",
    2: "Red_Hairy_Caterpillar",
    3: "Larva_Spodoptera",
}

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
# TEST = remainder (~0.10)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True,
                   help="Random seed for specimen-level shuffle")
    p.add_argument("--src",  type=str, default=str(SRC_DEFAULT),
                   help="Source soybean_yolo_clean directory")
    p.add_argument("--dst_root", type=str, default=str(DST_ROOT),
                   help="Root output directory")
    return p.parse_args()


def get_specimen_id(stem: str) -> str:
    """
    Extract specimen ID from image filename stem.

    Example:
      'Eocanthecona_Bug_A__00001IMG_00001_BURST20190912...'
       → 'Eocanthecona_Bug_A__00001'
    """
    parts = stem.split("__")
    if len(parts) < 2:
        # Fallback: use full stem as its own group
        return stem
    class_prefix  = parts[0]
    after_prefix   = parts[1]   # e.g. "00001IMG_00001_BURST..."
    specimen_num   = after_prefix.split("IMG")[0]   # e.g. "00001"
    return f"{class_prefix}__{specimen_num}"


def main():
    args   = parse_args()
    seed   = args.seed
    src    = Path(args.src)
    dst    = Path(args.dst_root) / f"seed{seed}_specimen"

    img_dir = src / "images"
    lbl_dir = src / "labels"

    # ── Gather all paired image/label files ───────────────────────────────────
    ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    all_images = sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in ext
        and (lbl_dir / (p.stem + ".txt")).exists()
    ])
    print(f"[seed={seed}_specimen] Found {len(all_images)} paired files.")

    # ── Group by specimen ID ───────────────────────────────────────────────────
    specimen_groups: dict[str, list[Path]] = defaultdict(list)
    for img_path in all_images:
        sid = get_specimen_id(img_path.stem)
        specimen_groups[sid].append(img_path)

    specimen_ids = sorted(specimen_groups.keys())
    print(f"  Unique specimens: {len(specimen_ids)}")

    # Show per-class specimen counts
    class_specimen_counts: dict[str, int] = defaultdict(int)
    for sid in specimen_ids:
        class_prefix = sid.split("__")[0]
        class_specimen_counts[class_prefix] += 1
    for cls, cnt in sorted(class_specimen_counts.items()):
        n_images = sum(len(specimen_groups[s]) for s in specimen_ids
                       if s.startswith(cls))
        print(f"    {cls}: {cnt} specimens, {n_images} images")

    # ── Shuffle and split at specimen level ───────────────────────────────────
    random.seed(seed)
    random.shuffle(specimen_ids)

    n_total   = len(specimen_ids)
    n_train   = int(n_total * TRAIN_RATIO)
    n_val     = int(n_total * VAL_RATIO)

    split_specimens = {
        "train": specimen_ids[:n_train],
        "val":   specimen_ids[n_train : n_train + n_val],
        "test":  specimen_ids[n_train + n_val :],
    }

    # ── Report split sizes ────────────────────────────────────────────────────
    print(f"\nSplit (specimen-level):")
    for split, sids in split_specimens.items():
        n_imgs = sum(len(specimen_groups[s]) for s in sids)
        print(f"  {split}: {len(sids)} specimens → {n_imgs} images")

    # ── Verify no specimen appears in multiple splits ─────────────────────────
    all_assigned = (set(split_specimens["train"]) |
                    set(split_specimens["val"])   |
                    set(split_specimens["test"]))
    assert len(all_assigned) == n_total, "BUG: some specimens unassigned"
    overlaps = (set(split_specimens["train"]) & set(split_specimens["val"]) |
                set(split_specimens["train"]) & set(split_specimens["test"]) |
                set(split_specimens["val"])   & set(split_specimens["test"]))
    assert len(overlaps) == 0, f"BUG: specimens in multiple splits: {overlaps}"
    print("  Overlap check passed — zero specimens shared across splits.")

    # ── Copy files ────────────────────────────────────────────────────────────
    for split, sids in split_specimens.items():
        img_out = dst / "images" / split
        lbl_out = dst / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for sid in sids:
            for img_path in specimen_groups[sid]:
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                shutil.copy2(img_path, img_out / img_path.name)
                shutil.copy2(lbl_path, lbl_out / (img_path.stem + ".txt"))

    print(f"\nFiles copied to: {dst}")

    # ── Write data.yaml ───────────────────────────────────────────────────────
    yaml_content = f"""\
path: {dst}
train: images/train
val:   images/val
test:  images/test

nc: {len(CLASS_NAMES)}
names:
"""
    for idx, name in CLASS_NAMES.items():
        yaml_content += f"  {idx}: {name}\n"

    (dst / "data.yaml").write_text(yaml_content)
    print(f"data.yaml written.")
    print(f"\nDone. To train on this split:")
    print(f"  python scripts/train_yolo_multiseed.py \\")
    print(f"      --model yolov8s --exp_name E3_yolov8s_specimen_seed{seed} \\")
    print(f"      --aug yolo_default --seed {seed} \\")
    print(f"      --data {dst}/data.yaml --device 0")


if __name__ == "__main__":
    main()

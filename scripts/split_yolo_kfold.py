#!/usr/bin/env python3
"""
split_yolo_kfold.py
-------------------
Creates 10-fold cross-validation splits at the SPECIMEN level for InsectBase.

HOW IT WORKS:
  1. Groups all 3807 images into 382 unique specimens using burst-photo naming convention
  2. Shuffles specimens with a fixed master seed (deterministic / reproducible)
  3. Divides into 10 equal folds of ~38-39 specimens each
  4. For each fold k (0-9):
       TEST  = fold  k              (~38 specimens, ~380 images)
       VAL   = fold (k+1) % 10     (~38 specimens, ~380 images)  [for YOLO early-stopping]
       TRAIN = all other 8 folds   (~306 specimens, ~3060 images)

  Zero overlap between train/val/test is guaranteed and checked.

USAGE (on Narval, from project root):
  # Create ALL 10 folds at once:
  python scripts/split_yolo_kfold.py --master_seed 42

  # Create a SINGLE fold (useful when you submit one job at a time):
  python scripts/split_yolo_kfold.py --master_seed 42 --fold 3

OUTPUT:
  soybean_yolo_splits/kfold_ms42/
    fold_0/   images/train/  images/val/  images/test/
              labels/train/  labels/val/  labels/test/
              data.yaml
    fold_1/   ...
    ...
    fold_9/   ...
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DEFAULT  = PROJECT_ROOT / "soybean_yolo_clean"
DST_ROOT     = PROJECT_ROOT / "soybean_yolo_splits"

CLASS_NAMES = {
    0: "Eocanthecona_Bug",
    1: "Tobacco_Caterpillar",
    2: "Red_Hairy_Caterpillar",
    3: "Larva_Spodoptera",
}

N_FOLDS = 10


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Specimen-level 10-fold CV split for InsectBase")
    p.add_argument("--master_seed", type=int, default=42,
                   help="Seed for specimen shuffle (controls fold assignment). Default: 42")
    p.add_argument("--fold", type=int, default=None,
                   help="If set, only generate this fold (0-9). Default: all 10 folds.")
    p.add_argument("--src",      type=str, default=str(SRC_DEFAULT),
                   help="Path to soybean_yolo_clean (flat images/ and labels/ dirs)")
    p.add_argument("--dst_root", type=str, default=str(DST_ROOT),
                   help="Parent directory for output splits")
    return p.parse_args()


def get_specimen_id(stem: str) -> str:
    """
    Extract specimen ID from image filename stem.
    Format: <CLASS>__<SPECIMEN_NUM>IMG_<FRAME_NUM>_BURST...
    Example: 'Eocanthecona_Bug__00001IMG_00001_BURST20180601...'
          -> 'Eocanthecona_Bug__00001'
    """
    parts = stem.split("__")
    if len(parts) < 2:
        return stem                           # fallback: treat whole stem as ID
    class_prefix  = parts[0]
    after_prefix  = parts[1]                  # e.g. '00001IMG_00001_BURST...'
    specimen_num  = after_prefix.split("IMG")[0]   # e.g. '00001'
    return f"{class_prefix}__{specimen_num}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    src      = Path(args.src)
    dst_root = Path(args.dst_root) / f"kfold_ms{args.master_seed}"
    img_dir  = src / "images"
    lbl_dir  = src / "labels"

    # 1. Collect all paired image/label files
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    all_images = sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in img_exts
        and (lbl_dir / (p.stem + ".txt")).exists()
    ])
    print(f"Found {len(all_images)} paired image/label files in {src}")

    # 2. Group by specimen ID
    specimen_groups: dict[str, list[Path]] = defaultdict(list)
    for img_path in all_images:
        sid = get_specimen_id(img_path.stem)
        specimen_groups[sid].append(img_path)

    specimen_ids = sorted(specimen_groups.keys())
    n_specimens  = len(specimen_ids)
    print(f"Unique specimens: {n_specimens}")

    # Per-class breakdown
    class_counts: dict[str, list] = defaultdict(list)
    for sid in specimen_ids:
        cls = sid.split("__")[0]
        class_counts[cls].append(sid)
    print("Per-class breakdown:")
    for cls, sids in sorted(class_counts.items()):
        n_imgs = sum(len(specimen_groups[s]) for s in sids)
        print(f"  {cls}: {len(sids)} specimens, {n_imgs} images")

    # 3. Shuffle specimens with master seed → deterministic fold assignment
    random.seed(args.master_seed)
    shuffled = specimen_ids.copy()
    random.shuffle(shuffled)

    # 4. Divide into N_FOLDS equal-ish groups
    folds: list[list[str]] = []
    fold_size = n_specimens // N_FOLDS
    remainder = n_specimens % N_FOLDS   # first `remainder` folds get +1 specimen
    idx = 0
    for k in range(N_FOLDS):
        size = fold_size + (1 if k < remainder else 0)
        folds.append(shuffled[idx: idx + size])
        idx += size

    print(f"\n10-Fold assignment summary (master_seed={args.master_seed}):")
    for k, fold in enumerate(folds):
        n_imgs = sum(len(specimen_groups[s]) for s in fold)
        print(f"  Fold {k:2d}: {len(fold):3d} specimens, {n_imgs:5d} images")

    # 5. Create directories for the requested fold(s)
    folds_to_create = list(range(N_FOLDS)) if args.fold is None else [args.fold]

    for k in folds_to_create:
        test_fold   = k
        val_fold    = (k + 1) % N_FOLDS
        train_folds = [i for i in range(N_FOLDS) if i != test_fold and i != val_fold]

        test_specimens  = folds[test_fold]
        val_specimens   = folds[val_fold]
        train_specimens = [s for i in train_folds for s in folds[i]]

        # --- Overlap check (should never fail, but verify) ---
        train_set = set(train_specimens)
        val_set   = set(val_specimens)
        test_set  = set(test_specimens)
        assert len(train_set & val_set)  == 0, f"Fold {k}: TRAIN/VAL overlap!"
        assert len(train_set & test_set) == 0, f"Fold {k}: TRAIN/TEST overlap!"
        assert len(val_set   & test_set) == 0, f"Fold {k}: VAL/TEST overlap!"
        total = len(train_set) + len(val_set) + len(test_set)
        assert total == n_specimens, f"Fold {k}: specimen count mismatch ({total} != {n_specimens})"

        n_train_imgs = sum(len(specimen_groups[s]) for s in train_specimens)
        n_val_imgs   = sum(len(specimen_groups[s]) for s in val_specimens)
        n_test_imgs  = sum(len(specimen_groups[s]) for s in test_specimens)

        print(f"\n--- Fold {k} ---")
        print(f"  TRAIN: {len(train_specimens):3d} specimens ({n_train_imgs} images)  "
              f"[folds {train_folds}]")
        print(f"  VAL:   {len(val_specimens):3d} specimens ({n_val_imgs} images)  "
              f"[fold {val_fold}]")
        print(f"  TEST:  {len(test_specimens):3d} specimens ({n_test_imgs} images)  "
              f"[fold {test_fold}]")
        print(f"  Overlap check passed ✓")

        dst = dst_root / f"fold_{k}"
        split_map = {
            "train": train_specimens,
            "val":   val_specimens,
            "test":  test_specimens,
        }

        for split, sids in split_map.items():
            img_out = dst / "images" / split
            lbl_out = dst / "labels" / split
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)
            for sid in sids:
                for img_path in specimen_groups[sid]:
                    lbl_path = lbl_dir / (img_path.stem + ".txt")
                    shutil.copy2(img_path, img_out / img_path.name)
                    shutil.copy2(lbl_path, lbl_out / (img_path.stem + ".txt"))

        # Write data.yaml
        names_block = "\n".join(f"  {i}: {n}" for i, n in CLASS_NAMES.items())
        yaml_text = (
            f"path: {dst}\n"
            f"train: images/train\n"
            f"val:   images/val\n"
            f"test:  images/test\n\n"
            f"nc: {len(CLASS_NAMES)}\n"
            f"names:\n{names_block}\n"
        )
        (dst / "data.yaml").write_text(yaml_text)
        print(f"  Directory created: {dst}")

    print(f"\nAll requested folds written to: {dst_root}")


if __name__ == "__main__":
    main()

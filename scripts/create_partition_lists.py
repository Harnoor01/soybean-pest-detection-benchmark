from pathlib import Path
import csv
import re

ROOT = Path("soybean_yolo_splits")
OUT = Path("partition_lists")
OUT.mkdir(exist_ok=True)


def get_specimen_id(filename: str) -> str:
    stem = Path(filename).stem

    match = re.search(r"(\d+)IMG", stem)
    if match:
        return match.group(1)

    return stem


def export_random_split(folder_name: str):
    base = ROOT / folder_name
    output_file = OUT / f"{folder_name}.csv"

    rows = []

    for split in ["train", "val", "test"]:
        image_dir = base / "images" / split

        if not image_dir.exists():
            print(f"Missing: {image_dir}")
            continue

        for image_path in sorted(image_dir.iterdir()):
            if image_path.is_file():
                rows.append({
                    "filename": image_path.name,
                    "specimen_id": get_specimen_id(image_path.name),
                    "split": split
                })

    with output_file.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["filename", "specimen_id", "split"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created {output_file} with {len(rows)} assignments")


def export_kfold():
    base = ROOT / "kfold_ms42"
    output_file = OUT / "kfold_ms42.csv"

    rows = []

    for fold_dir in sorted(base.glob("fold_*")):
        fold_name = fold_dir.name

        for split in ["train", "val", "test"]:
            image_dir = fold_dir / "images" / split

            if not image_dir.exists():
                print(f"Missing: {image_dir}")
                continue

            for image_path in sorted(image_dir.iterdir()):
                if image_path.is_file():
                    rows.append({
                        "filename": image_path.name,
                        "specimen_id": get_specimen_id(image_path.name),
                        "fold": fold_name,
                        "split": split
                    })

    with output_file.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["filename", "specimen_id", "fold", "split"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created {output_file} with {len(rows)} assignments")


random_split_folders = [
    "seed0",
    "seed42",
    "seed123",
    "seed0_specimen",
    "seed42_specimen",
    "seed123_specimen"
]

for folder in random_split_folders:
    export_random_split(folder)

export_kfold()
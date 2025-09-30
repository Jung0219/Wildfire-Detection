import os
import json
import shutil
from pathlib import Path

# ==== CONFIGURATION ====
PARENT_DIR = "/lab/projects/fire_smoke_awr/data/classification/training/E+neither_A"   # contains "images" + "labels.json"
OUTPUT_DIR = os.path.join(PARENT_DIR, "ImageFolder")      # YOLO-style dataset will be written here
# ========================

def convert_to_yolo_cls_format(parent_dir, output_dir):
    parent_dir = Path(parent_dir)
    output_dir = Path(output_dir)
    images_dir = parent_dir / "images"
    labels_file = parent_dir / "labels.json"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found at {images_dir}")
    if not labels_file.exists():
        raise FileNotFoundError(f"labels.json not found at {labels_file}")

    # Load labels
    with open(labels_file, "r") as f:
        labels = json.load(f)

    # Create output structure (only train/)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subfolders for classes
    classes = sorted(set(labels.values()))
    for c in classes:
        (output_dir / c).mkdir(parents=True, exist_ok=True)

    # Copy images into class folders
    for img_name, cls in labels.items():
        src = images_dir / img_name
        dst = output_dir / cls / img_name
        if not src.exists():
            print(f"Warning: {src} not found, skipping")
            continue
        shutil.copy(src, dst)

    print(f"Conversion complete. YOLO dataset at {output_dir}")
    print(f"Classes: {classes}")

if __name__ == "__main__":
    convert_to_yolo_cls_format(PARENT_DIR, OUTPUT_DIR)

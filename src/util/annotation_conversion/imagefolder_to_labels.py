import os
import json
import shutil
from pathlib import Path

# ==== CONFIGURATION ====
SRC_DIR = "/lab/projects/fire_smoke_awr/data/classification/test_sets/BCDE_val_composite_inference_test_set/imagefolder"
# List of class folders to include
CLASS_FOLDERS = ["foreground", "background"]   # <-- specify exactly which subfolders to use
# Specify the parent directory where images/ and labels.json will be saved
DST_PARENT_DIR = "/lab/projects/fire_smoke_awr/data/classification/test_sets/BCDE_val_composite_inference_test_set"

DST_IMG_DIR = str(Path(DST_PARENT_DIR) / "images")
DST_LABELS = str(Path(DST_PARENT_DIR) / "labels.json")
IMG_EXTS = [".jpg", ".jpeg", ".png"]
# ========================

def main():
    os.makedirs(DST_IMG_DIR, exist_ok=True)
    labels = {}

    for class_name in CLASS_FOLDERS:
        class_dir = Path(SRC_DIR) / class_name
        if not class_dir.is_dir():
            print(f"Warning: {class_dir} not found, skipping.")
            continue

        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() not in IMG_EXTS:
                continue

            # Prefix with class name to avoid collisions
            new_name = f"{class_name}_{img_file.name}"
            dst_path = Path(DST_IMG_DIR) / new_name

            shutil.copy(img_file, dst_path)
            labels[new_name] = class_name

    with open(DST_LABELS, "w") as f:
        json.dump(labels, f, indent=4)

    print(f"Saved {len(labels)} labels to {DST_LABELS}")

if __name__ == "__main__":
    main()

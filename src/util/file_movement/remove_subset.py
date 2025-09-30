import os
import shutil
from pathlib import Path

# ========== CONFIG ==========
PARENT_DIR = "/lab/projects/fire_smoke_awr/data/detection/datasets/data_mining/single_objects_lt_130"   # e.g., /lab/projects/.../all_data
CHILD_DIR  = "/lab/projects/fire_smoke_awr/data/detection/datasets/data_mining/single_objects_lt_130/early_fire"    # e.g., /lab/projects/.../subset_to_remove
OUTPUT_DIR = "/lab/projects/fire_smoke_awr/data/detection/datasets/data_mining/single_objects_lt_130/classifier_fp_mining"   # new folder for "parent minus child"
# ============================

def main():
    parent_images = Path(PARENT_DIR) / "images/test"
    parent_labels = Path(PARENT_DIR) / "labels/test"
    child_images = Path(CHILD_DIR) / "images/test"
    child_labels = Path(CHILD_DIR) / "labels/test"

    out_images = Path(OUTPUT_DIR) / "images/test"
    out_labels = Path(OUTPUT_DIR) / "labels/test"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # collect child basenames (stems)
    child_stems = {p.stem for p in child_images.iterdir() if p.is_file()}

    kept = 0
    removed = 0

    for img_file in parent_images.iterdir():
        if img_file.stem in child_stems:
            removed += 1
            continue
        # copy image
        shutil.copy2(img_file, out_images / img_file.name)
        # copy matching label if exists
        lbl_file = parent_labels / (img_file.stem + ".txt")
        if lbl_file.exists():
            shutil.copy2(lbl_file, out_labels / lbl_file.name)
        kept += 1

    print(f"[INFO] Done. Kept {kept}, removed {removed}.")

if __name__ == "__main__":
    main()

import os
from pathlib import Path

# ==== CONFIG ====
GT_DIR   = "/lab/projects/fire_smoke_awr/data/training/deduplicated/phash10/B+C+D+E/split/images/test"   # folder with images
PRED_DIR = "/lab/projects/fire_smoke_awr/outputs/yolo/baseline/B+C+D+E_10%/labels"      # folder with YOLO .txt predictions
# ================

# Collect all image stems from ground truth
img_exts = {".jpg", ".jpeg", ".png"}
img_stems = [Path(f).stem for f in os.listdir(GT_DIR) if Path(f).suffix.lower() in img_exts]

# Ensure prediction dir exists
os.makedirs(PRED_DIR, exist_ok=True)

missing = []
for stem in img_stems:
    pred_file = Path(PRED_DIR) / f"{stem}.txt"
    if not pred_file.exists():
        open(pred_file, "w").close()  # create empty
        missing.append(pred_file)

print(f"[INFO] Checked {len(img_stems)} images.")
print(f"[INFO] Created {len(missing)} empty prediction files in {PRED_DIR}.")

import os
import json
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path

# ==== CONFIG ====
PARENT_DIR = Path("/lab/projects/fire_smoke_awr/outputs/yolo/classification/ABCDE_early_fire_removed")
IMG_DIR = Path("/lab/projects/fire_smoke_awr/data/classification/test_sets/EF_dev")
SAVE_DIR = IMG_DIR.name

MODEL_PATH = PARENT_DIR / "train" / "weights" / "best.pt"
SAVE_JSON = PARENT_DIR / SAVE_DIR / "preds.json"
# ================

# load model
model = YOLO(str(MODEL_PATH))

# Hard code class mapping: 0 -> "background", 1 & 2 -> "foreground"
class_map = {0: "background", 1: "foreground", 2: "foreground"}

print("\nFinal class mapping:")
for idx, default_name in model.names.items():
    mapped = class_map.get(idx, default_name)
    print(f"class {idx}: {default_name} --> {mapped}")


# collect all images
img_files = []
for root, _, files in os.walk(IMG_DIR):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            img_files.append(os.path.join(root, f))

print(f"\nFound {len(img_files)} images in {IMG_DIR}")

# run inference
pred_dict = {}
for img_path in tqdm(img_files, desc="Running inference"):
    results = model.predict(img_path, verbose=False)
    probs = results[0].probs

    cls_id = int(probs.top1)         # predicted class index
    cls_name = class_map[cls_id]     # mapped class name

    fname = os.path.basename(img_path)
    pred_dict[fname] = cls_name

# save predictions
os.makedirs(os.path.dirname(SAVE_JSON), exist_ok=True)
with open(SAVE_JSON, "w") as f:
    json.dump(pred_dict, f, indent=2)

print(f"\nSaved predictions for {len(pred_dict)} images to {SAVE_JSON}")

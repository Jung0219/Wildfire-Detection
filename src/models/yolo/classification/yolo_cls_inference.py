import os
import json
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path

# ==== CONFIG ====
MODEL_DIR = Path("/lab/projects/fire_smoke_awr/outputs/yolo/classification/AD_phash3_early_smoke")
IMG_DIR = Path("/lab/projects/fire_smoke_awr/data/classification/training/AD_phash3_early_smoke/test")
SAVE_DIR = IMG_DIR.name

MODEL_PATH = MODEL_DIR / "train" / "weights" / "best.pt"
SAVE_JSON = MODEL_DIR / SAVE_DIR / "preds.json"
# ================

# load model
model = YOLO(str(MODEL_PATH))

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

    cls_id = int(probs.top1)
    cls_name = "background" if cls_id == 0 else "foreground"

    fname = os.path.basename(img_path)
    pred_dict[fname] = cls_name

# save predictions
os.makedirs(os.path.dirname(SAVE_JSON), exist_ok=True)
with open(SAVE_JSON, "w") as f:
    json.dump(pred_dict, f, indent=2)

print(f"\nSaved predictions for {len(pred_dict)} images to {SAVE_JSON}")

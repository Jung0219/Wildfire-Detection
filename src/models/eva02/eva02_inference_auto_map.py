import os
import sys
import shutil
import json
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from eva02_model import EVA02Classifier   # Letterbox/CenterPadOrLetterbox already defined here

# ============ CONFIG ============
PARENT_DIR      = "/lab/projects/fire_smoke_awr/outputs/eva02/fp_mined_v2_sgd"
# test set (ImageFolder expects subdirs per class)
DATA_DIR        = "/lab/projects/fire_smoke_awr/data/classification/test_sets/EF_dev"
SAVE_SUBDIR     =  os.path.basename(DATA_DIR)

CLASSES         = ["background", "fire", "smoke"]
NUM_CLASSES     = len(CLASSES)

TRANSFORM_MODE  = "letterbox"   # choose: "letterbox" or "centerpad"
IMG_SIZE        = 224
BATCH_SIZE      = 128
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
# =================================

MODEL_PATH = os.path.join(PARENT_DIR, "train", "weights", "best_loss.pt")
SAVE_DIR   = os.path.join(PARENT_DIR, SAVE_SUBDIR)

# handle existing save dir
if os.path.exists(SAVE_DIR):
    ans = input(f"[WARN] Directory {SAVE_DIR} already exists. Overwrite it? [y/N]: ").strip().lower()
    if ans not in ("y", "yes"):
        print("Aborting.")
        sys.exit(0)
    else:
        shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- Dataset + Loader ----------
model = EVA02Classifier(
    num_classes=NUM_CLASSES,
    pretrained=False,
    transform=TRANSFORM_MODE,
    img_size=IMG_SIZE
)

transform = model.get_transform()
test_ds = datasets.ImageFolder(root=DATA_DIR, transform=transform)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ---------- Load weights ----------
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model = model.to(DEVICE).eval()

# ---------- Inference ----------
results = {}
with torch.no_grad():
    for batch_idx, (imgs, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        # remap ternary (0,1,2) -> binary (0=background, 1=foreground)
        preds = [0 if p == 0 else 1 for p in preds]

        start = batch_idx * test_loader.batch_size
        end   = start + len(imgs)
        batch_indices = range(start, end)

        for idx, pred in zip(batch_indices, preds):
            fname, _ = test_ds.samples[idx]
            results[os.path.basename(fname)] = "foreground" if pred == 1 else "background"

# ---------- Save ----------
save_path = os.path.join(SAVE_DIR, "preds.json")
with open(save_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"[{TRANSFORM_MODE}] Saved per-image predictions (binary foreground/background) to {save_path}")


import os
import sys
import yaml
import pandas as pd
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from eva02_model import EVA02Classifier
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ================= Load Config =================
if len(sys.argv) < 2:
    print("Usage: python eva02_train.py <config.yaml>")
    sys.exit(1)

config_path = sys.argv[1]
with open(config_path, "r") as f:
    CONFIG = yaml.safe_load(f)

DATA_DIR       = CONFIG["DATA_DIR"]
PARENT_DIR     = CONFIG["PARENT_DIR"]
SAVE_SUBDIR    = CONFIG["SAVE_SUBDIR"]

BATCH_SIZE     = CONFIG["BATCH_SIZE"]
EPOCHS         = CONFIG["EPOCHS"]
LR             = float(CONFIG["LR"])
MODEL_NAME     = CONFIG["MODEL_NAME"]
IMG_SIZE       = CONFIG["IMG_SIZE"]
TRANSFORM_MODE = CONFIG.get("TRANSFORM_MODE", "letterbox")
DEVICE         = CONFIG.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
TEST           = CONFIG.get("TEST", False)


# ---------- Save directory ----------
SAVE_DIR   = os.path.join(PARENT_DIR, SAVE_SUBDIR)
MODEL_DIR = os.path.join(SAVE_DIR, "weights")
os.makedirs(MODEL_DIR, exist_ok=True)

BEST_ACC_PATH  = os.path.join(MODEL_DIR, "best_acc.pt")
BEST_LOSS_PATH = os.path.join(MODEL_DIR, "best_loss.pt")
LAST_PATH      = os.path.join(MODEL_DIR, "last.pt")

HISTORY_PATH = os.path.join(SAVE_DIR, "training_history.csv")

# create header if not exists
if not os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "precision", "recall", "val_acc", "lr"])


os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_PATH = os.path.join(SAVE_DIR, "weights", "best.pt")

# ---------- Save config copy ----------
with open(os.path.join(SAVE_DIR, "config_used.yaml"), "w") as f:
    yaml.dump(CONFIG, f, default_flow_style=False)

print(f"[INFO] Loaded config from {config_path}")
print(f"[INFO] Saving run artifacts to {SAVE_DIR}")

# ---------- Build model ----------
model = EVA02Classifier(
    model_name=MODEL_NAME,
    num_classes=None,
    pretrained=True,
    transform=TRANSFORM_MODE,
    img_size=IMG_SIZE
).to(DEVICE)
transform = model.get_transform()

# ---------- Build datasets ----------
def make_dataset(split):
    img_dir = os.path.join(DATA_DIR, split)
    return datasets.ImageFolder(root=img_dir, transform=transform)

train_ds = make_dataset("train")
val_ds   = make_dataset("val")
if TEST:
    test_ds  = make_dataset("test")

classes = train_ds.classes
NUM_CLASSES = len(classes)
print("Training class order:", classes)

# Update model with correct number of classes
model = EVA02Classifier(
    model_name=MODEL_NAME,
    num_classes=NUM_CLASSES,
    pretrained=True,
    transform=TRANSFORM_MODE,
    img_size=IMG_SIZE
).to(DEVICE)
transform = model.get_transform()

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
if TEST:
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ---------- Loss & Optimizer ----------
criterion = nn.CrossEntropyLoss()

opt_name = CONFIG.get("OPTIMIZER", "adamw").lower()
weight_decay = CONFIG.get("WEIGHT_DECAY", 0.0)

if opt_name == "adamw":
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)
elif opt_name == "adam":
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
elif opt_name == "sgd":
    momentum = CONFIG.get("MOMENTUM", 0.9)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=momentum,
                          weight_decay=weight_decay, nesterov=True)
else:
    raise ValueError(f"Unsupported optimizer: {opt_name}")

print(f"[INFO] Using optimizer: {opt_name.upper()} (lr={LR}, weight_decay={weight_decay})")


# ---------- Scheduler ----------
scheduler = None
sched_name = CONFIG.get("SCHEDULER", "none").lower()
if sched_name == "cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
elif sched_name == "step":
    step_size = CONFIG.get("STEP_SIZE", 30)
    gamma = CONFIG.get("GAMMA", 0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
elif sched_name == "none":
    scheduler = None
else:
    raise ValueError(f"Unsupported scheduler: {sched_name}")

print(f"[INFO] Using scheduler: {sched_name.upper()}")


# ---------- Training ----------
best_acc = 0.0
best_val_loss = float("inf")

train_losses, val_losses = [], []
precisions, recalls = [], []

for epoch in range(EPOCHS):
    # Training loop
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validation loop
    model.eval()
    running_val_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    val_loss = running_val_loss / len(val_loader)
    val_losses.append(val_loss)

    prec, rec, _, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    precisions.append(prec)
    recalls.append(rec)

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Epoch {epoch+1}/{EPOCHS} - "
          f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {acc:.4f}")

    current_lr = optimizer.param_groups[0]["lr"]

    # log metrics immediately
    with open(HISTORY_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, train_loss, val_loss, prec, rec, acc, current_lr])

    # step scheduler
    if scheduler:
        scheduler.step()

    # save best accuracy
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), BEST_ACC_PATH)
        print(f"  >> Saved best_acc model with acc {best_acc:.4f}")

    # save best validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), BEST_LOSS_PATH)
        print(f"  >> Saved best_loss model with val_loss {best_val_loss:.4f}")

    # always save last epoch
    torch.save(model.state_dict(), LAST_PATH)
    print(f"  >> Saved last model at epoch {epoch+1}")

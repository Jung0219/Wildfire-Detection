#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Learn optimal alpha for weighted average fusion:
final_conf = alpha * det_conf + (1 - alpha) * cls_conf
Trains alpha using BCE loss on CSV data.
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, accuracy_score

# === CONFIG ===
CSV_PATH = "/lab/projects/fire_smoke_awr/src/pipeline/2-stage/A_phash10_single_objects_pred_lt_100.csv"
LR       = 0.05
EPOCHS   = 2000
# ==============

# Load CSV
df = pd.read_csv(CSV_PATH)
det_conf = torch.tensor(df["det_conf"].values, dtype=torch.float32)
cls_conf = torch.tensor(df["cls_conf"].values, dtype=torch.float32)
y_true   = torch.tensor(df["label"].values, dtype=torch.float32)

# Learnable parameter
alpha = torch.nn.Parameter(torch.tensor(0.5))
optimizer = optim.Adam([alpha], lr=LR)
loss_fn = nn.BCELoss()

for epoch in range(EPOCHS):
    final_conf = alpha * det_conf + (1 - alpha) * cls_conf
    final_conf = final_conf.clamp(1e-6, 1 - 1e-6)  # avoid log(0)
    loss = loss_fn(final_conf, y_true)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # constrain alpha to [0,1]
    with torch.no_grad():
        alpha.clamp_(0.0, 1.0)

print(f"✅ Optimal alpha: {alpha.item():.4f}")

# --- Evaluate ---
final_conf = (alpha * det_conf + (1 - alpha) * cls_conf).detach().numpy()
y_true_np  = y_true.numpy()

print("=== Metrics with optimal alpha ===")
print(f"Brier score: {brier_score_loss(y_true_np, final_conf):.4f}")
print(f"Log-loss:    {log_loss(y_true_np, final_conf):.4f}")
print(f"ROC AUC:     {roc_auc_score(y_true_np, final_conf):.4f}")
print(f"Accuracy:    {accuracy_score(y_true_np, final_conf >= 0.5):.4f}")

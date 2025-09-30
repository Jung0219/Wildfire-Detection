#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calibration analysis for binary classifier outputs.
Reads CSV with [image, det_conf, cls_conf, temp_conf, logreg_conf, label].
Computes multiple metrics:
- ECE (Expected Calibration Error)
- Brier score
- Log-loss
- ROC AUC
- Accuracy (threshold=0.5)
And plots reliability diagram.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, accuracy_score

# === CONFIG ===
CSV_PATH = "/lab/projects/fire_smoke_awr/src/pipeline/2-stage/A_phash10_single_objects_pred_lt_100.csv"
N_BINS   = 15
# ==============

# Load CSV
df = pd.read_csv(CSV_PATH)

# true labels (0=background, 1=fire/smoke)
y_true = df["label"].values.astype(int)

# predicted probabilities for class 1 (fire/smoke)
y_prob = df["cls_conf"].values.astype(float)

# --- Metrics ---
brier = brier_score_loss(y_true, y_prob)
logloss = log_loss(y_true, y_prob)
auc = roc_auc_score(y_true, y_prob)
acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))

# --- ECE ---
bins = np.linspace(0.0, 1.0, N_BINS + 1)
binids = np.digitize(y_prob, bins) - 1
ece = 0.0
accs, confs = [], []
for i in range(N_BINS):
    mask = binids == i
    if np.sum(mask) > 0:
        acc_bin = np.mean(y_true[mask] == (y_prob[mask] >= 0.5))
        conf_bin = np.mean(y_prob[mask])
        n = np.sum(mask)
        accs.append(acc_bin)
        confs.append(conf_bin)
        ece += np.abs(acc_bin - conf_bin) * (n / len(y_true))

# Print results
print("=== Calibration & Performance Metrics ===")
print(f"Brier score: {brier:.4f}")
print(f"Log-loss:    {logloss:.4f}")
print(f"ROC AUC:     {auc:.4f}")
print(f"Accuracy:    {acc:.4f}")
print(f"ECE:         {ece:.4f}")

# --- Reliability diagram ---
plt.figure(figsize=(6,6))
plt.plot([0,1], [0,1], "k--", label="Perfect calibration")
plt.bar(bins[:-1], accs, width=1/N_BINS, alpha=0.5, edgecolor="black", label="Accuracy per bin")
plt.plot(np.array(confs), accs, "o-", color="red", label="Reliability curve")
plt.xlabel("Predicted confidence")
plt.ylabel("Empirical accuracy")
plt.title("Reliability Diagram")
plt.legend()
plt.tight_layout()
out_path = CSV_PATH.replace(".csv", "_reliability.png")
plt.savefig(out_path, dpi=300)
print(f"Saved reliability diagram to {out_path}")

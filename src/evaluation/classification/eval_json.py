import os
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================
CONFIG = {
    "json_file": "/lab/projects/fire_smoke_awr/outputs/eva02/fp_mined_v2/EF_dev/preds.json",  # Path to predictions JSON
    "gt_dir": "/lab/projects/fire_smoke_awr/data/classification/test_sets/EF_dev",          # Ground truth folder (organized by class)
    "normalize_confusion": True,               # Save normalized confusion matrix
    "figsize": (6, 5),                         # Figure size for plots
    "cmap": "Blues",                           # Color map for heatmaps
    "dpi": 150                                 # DPI for saving figures
}
# =========================


def evaluate_predictions(json_file, imagefolder):
    # Load predictions
    with open(json_file, "r") as f:
        preds = json.load(f)

    y_true, y_pred = [], []
    class_counts = Counter()
    class_correct = Counter()

    # Wrap iteration with tqdm
    for img_name, pred_class in tqdm(preds.items(), desc="Evaluating predictions"):
        found = False
        for root, dirs, files in os.walk(imagefolder):
            if img_name in files:
                found = True
                gt_class = os.path.basename(root)
                y_true.append(gt_class)
                y_pred.append(pred_class)

                class_counts[gt_class] += 1
                if gt_class == pred_class:
                    class_correct[gt_class] += 1
                break
        if not found:
            print(f"Warning: {img_name} not found in {imagefolder}")

    total = len(y_true)
    correct = sum(gt == pr for gt, pr in zip(y_true, y_pred))

    print(f"Total images evaluated: {total}")
    print(f"Overall accuracy: {correct / total:.4f}" if total > 0 else "No images evaluated")
    print("\nPer-class accuracy:")
    for c in class_counts:
        acc = class_correct[c] / class_counts[c]
        print(f"  {c}: {acc:.4f} ({class_correct[c]}/{class_counts[c]})")

    if total > 0:
        save_confusion_matrices(y_true, y_pred, json_file)


def save_confusion_matrices(y_true, y_pred, json_file):
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # === Raw confusion matrix ===
    fig, ax = plt.subplots(figsize=CONFIG["figsize"], dpi=CONFIG["dpi"])
    sns.heatmap(cm, annot=True, fmt="d", cmap=CONFIG["cmap"],
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix (Counts)")
    raw_path = os.path.join(os.path.dirname(json_file), "confusion_matrix_counts.png")
    plt.tight_layout()
    plt.savefig(raw_path)
    plt.close()
    print(f"Saved raw confusion matrix: {raw_path}")

    # === Normalized confusion matrix ===
    if CONFIG["normalize_confusion"]:
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=CONFIG["figsize"], dpi=CONFIG["dpi"])
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap=CONFIG["cmap"],
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        ax.set_title("Confusion Matrix (Normalized)")
        norm_path = os.path.join(os.path.dirname(json_file), "confusion_matrix_normalized.png")
        plt.tight_layout()
        plt.savefig(norm_path)
        plt.close()
        print(f"Saved normalized confusion matrix: {norm_path}")


if __name__ == "__main__":
    evaluate_predictions(CONFIG["json_file"], CONFIG["gt_dir"])

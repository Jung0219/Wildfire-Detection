import os
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ==== CONFIGURATION ====
PRED_DIR = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/baseline/A_phash10_single_objects/composites"
PRED_PARENT = str(Path(PRED_DIR).parent)
PRED_BASENAME = Path(PRED_DIR).name
OUTPUT_PLOT = os.path.join(PRED_PARENT, f"{PRED_BASENAME}_obj_size_vs_conf.png")
IMG_SIZE = 640
NUM_BINS = 30  # number of bins for smoothing
# ========================

def parse_yolo_file(file_path, img_size=640):
    sizes, confs = [], []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            # YOLO format: class x y w h conf
            _, _, _, w, h, conf = map(float, parts)
            obj_size = math.sqrt(w * h) * img_size
            sizes.append(obj_size)
            confs.append(conf)
    return sizes, confs

def main():
    all_sizes, all_confs = [], []

    for fname in os.listdir(PRED_DIR):
        if fname.endswith(".txt"):
            file_path = os.path.join(PRED_DIR, fname)
            sizes, confs = parse_yolo_file(file_path, IMG_SIZE)
            all_sizes.extend(sizes)
            all_confs.extend(confs)

    if not all_sizes:
        print("No predictions found.")
        return

    # Scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(all_sizes, all_confs, alpha=0.3, s=10, label="Predictions")

    # Bin average trend
    sizes_arr = np.array(all_sizes)
    confs_arr = np.array(all_confs)
    bins = np.linspace(sizes_arr.min(), sizes_arr.max(), NUM_BINS+1)
    bin_idx = np.digitize(sizes_arr, bins)
    bin_means = [confs_arr[bin_idx == i].mean() if (bin_idx == i).any() else np.nan 
                 for i in range(1, len(bins))]
    bin_centers = (bins[:-1] + bins[1:]) / 2

    plt.plot(bin_centers, bin_means, color="red", linewidth=2, label="Trend (bin avg)")

    plt.xlabel("Object size (sqrt(w*h) * 640)")
    plt.ylabel("Confidence")
    plt.title("Object Size vs Confidence")
    plt.grid(True)
    plt.legend()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    plt.show()
    print(f"Saved plot with trend to {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()

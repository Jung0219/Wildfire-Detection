"""Measure runtime for a regular YOLO detector run.

Usage:
    python -m src.runtime_measurement.measure_yolo
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO

# ================= CONFIG (edit me) =================
IMAGE_DIR = Path("/lab/projects/fire_smoke_awr/data/final/test_data/small_object_set/hand-filtered/composites/images")
MODEL_PATH = Path("/lab/projects/fire_smoke_awr/weights/yolo11n.pt")
OUTPUT_DIR = Path("/lab/projects/fire_smoke_awr/outputs/runtime_measurement/runs/regular_yolo")
DEVICE = "cuda:5"
IMG_SIZE = 640
CONF_THRESH = 0.001
USE_HALF = False
WARMUP_ITERS = 2
# ====================================================


@dataclass
class TimingRow:
    image: str
    load_ms: float
    infer_ms: float
    total_ms: float


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    k = (len(vals) - 1) * pct / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    return vals[f] + (vals[c] - vals[f]) * (k - f)


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize(rows: Iterable[TimingRow]) -> dict:
    infer_vals = [r.infer_ms for r in rows]
    total_vals = [r.total_ms for r in rows]
    return {
        "infer_mean_ms": sum(infer_vals) / len(infer_vals) if infer_vals else 0.0,
        "infer_p50_ms": percentile(infer_vals, 50),
        "infer_p90_ms": percentile(infer_vals, 90),
        "infer_p99_ms": percentile(infer_vals, 99),
        "total_mean_ms": sum(total_vals) / len(total_vals) if total_vals else 0.0,
        "total_p50_ms": percentile(total_vals, 50),
        "total_p90_ms": percentile(total_vals, 90),
        "total_p99_ms": percentile(total_vals, 99),
        "throughput_imgs_per_s": (len(total_vals) / (sum(total_vals) / 1000)) if total_vals else 0.0,
    }


def load_images(image_dir: Path) -> list[Path]:
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])


def main() -> None:
    print("Runtime measurement: regular YOLO")
    print(f"Model: {MODEL_PATH}")
    print(f"Images: {IMAGE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)
    images = load_images(IMAGE_DIR)
    if not images:
        raise ValueError(f"No images found in {IMAGE_DIR}")

    # Warm-up
    with torch.inference_mode():
        for img_path in images[:WARMUP_ITERS]:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            sync_cuda()
            model.predict(image, imgsz=IMG_SIZE, conf=CONF_THRESH, device=DEVICE, half=USE_HALF, verbose=False)
            sync_cuda()

    rows: list[TimingRow] = []
    with torch.inference_mode():
        for img_path in tqdm(images, desc="Measuring"):
            t0 = time.perf_counter()
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            t1 = time.perf_counter()

            sync_cuda()
            model.predict(image, imgsz=IMG_SIZE, conf=CONF_THRESH, device=DEVICE, half=USE_HALF, verbose=False)
            sync_cuda()
            t2 = time.perf_counter()

            load_ms = (t1 - t0) * 1000
            infer_ms = (t2 - t1) * 1000
            total_ms = (t2 - t0) * 1000
            rows.append(TimingRow(img_path.name, load_ms, infer_ms, total_ms))

    summary = summarize(rows)

    csv_path = OUTPUT_DIR / "timings.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "load_ms", "infer_ms", "total_ms"])
        for r in rows:
            writer.writerow([r.image, f"{r.load_ms:.3f}", f"{r.infer_ms:.3f}", f"{r.total_ms:.3f}"])

    print(f"Wrote per-image timings to: {csv_path}")
    print("Summary (ms):")
    for key, val in summary.items():
        print(f"  {key}: {val:.3f}")


if __name__ == "__main__":
    main()

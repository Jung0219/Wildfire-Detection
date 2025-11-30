"""Measure runtime for the composite-only detector pipeline (no classifier).

Usage:
    python -m src.runtime_measurement.measure_composite_detector
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
import yaml
from tqdm import tqdm

from src.full_pipeline.config.config import MRClassifierConfig, ensure_dir
from src.full_pipeline.data.loader import base_name as basename_from_path, load_image
from src.full_pipeline.models.detector import load_detector
from src.full_pipeline.postprocess.mapping import map_detections
from src.full_pipeline.postprocess.nms import apply_nms
from src.full_pipeline.preprocess.composite import prepare_image_for_detection

# ================= CONFIG (edit me) =================
CONFIG_PATH = Path("/lab/projects/fire_smoke_awr/src/full_pipeline/run/config.yaml")
OUTPUT_DIR = Path("/lab/projects/fire_smoke_awr/outputs/runtime_measurement/runs/composite_detector")
DEVICE = "cuda:5"
SAVE_LABELS = False
SAVE_COMPOSITES = False
USE_HALF = False
WARMUP_ITERS = 2
# ====================================================


@dataclass
class TimingRow:
    image: str
    load_ms: float
    composite_ms: float
    infer_ms: float
    post_ms: float
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


def load_cfg(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def collect_images(cfg_dict: dict, cfg: MRClassifierConfig) -> list[Path]:
    image_paths = []
    if cfg_dict.get("image_path"):
        image_paths.append(Path(cfg_dict["image_path"]))
    if cfg_dict.get("image"):
        image_paths.append(Path(cfg_dict["image"]))
    if not image_paths and cfg_dict.get("image_dir"):
        image_dir = Path(cfg_dict["image_dir"])
        image_paths.extend([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not image_paths:
        # Fallback to default split layout
        image_dir = cfg.image_dir / "images" / "test" if (cfg.image_dir / "images" / "test").exists() else cfg.image_dir
        image_paths.extend([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    return sorted(image_paths)


def main() -> None:
    print("Runtime measurement: composite detector (no classifier)")
    print(f"Config: {CONFIG_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg_dict = load_cfg(CONFIG_PATH)
    cfg = MRClassifierConfig(
        image_dir=Path(cfg_dict.get("gt_dir", Path(cfg_dict.get("image_dir", MRClassifierConfig().image_dir)))),
        output_dir=OUTPUT_DIR,
        detector_weights=Path(cfg_dict.get("detector_weights", MRClassifierConfig().detector_weights)),
        intermediate_size=int(cfg_dict.get("intermediate_size", MRClassifierConfig().intermediate_size)),
        nms_iou_thresh=float(cfg_dict.get("nms_iou_thresh", MRClassifierConfig().nms_iou_thresh)),
        anchor_y_frac=float(cfg_dict.get("anchor_y_frac", MRClassifierConfig().anchor_y_frac)),
        save_debug=False,
        save_composites=SAVE_COMPOSITES,
        device=cfg_dict.get("device", DEVICE),
    )

    detector = load_detector(cfg.detector_weights, device=cfg.device)
    images = collect_images(cfg_dict, cfg)
    if not images:
        raise ValueError("No images found for measurement.")

    # Persist the effective config for reproducibility
    ensure_dir(cfg.output_dir)
    saved_cfg_path = cfg.output_dir / "config_used.yaml"
    with open(saved_cfg_path, "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Warm-up
    with torch.inference_mode():
        for img_path in images[:WARMUP_ITERS]:
            image = load_image(img_path)
            composite, _ = prepare_image_for_detection(image, cfg.intermediate_size, cfg.anchor_y_frac)
            sync_cuda()
            detector.predict(composite, imgsz=640, conf=0.001, device=cfg.device, half=USE_HALF, verbose=False)
            sync_cuda()

    rows: list[TimingRow] = []
    with torch.inference_mode():
        for img_path in tqdm(images, desc="Measuring"):
            t0 = time.perf_counter()
            image = load_image(img_path)
            base = basename_from_path(img_path)
            t1 = time.perf_counter()

            composite, meta = prepare_image_for_detection(image, cfg.intermediate_size, cfg.anchor_y_frac)
            t2 = time.perf_counter()

            sync_cuda()
            yolo_res = detector.predict(composite, imgsz=640, conf=0.001, device=cfg.device, half=USE_HALF, verbose=False)[0]
            sync_cuda()
            t3 = time.perf_counter()

            mapped = map_detections(yolo_res, meta, image.shape)
            if cfg.nms_iou_thresh and cfg.nms_iou_thresh > 0:
                final_dets = apply_nms(mapped, cfg.nms_iou_thresh)
            else:
                final_dets = [[d[0], d[1], d[2], d[3], d[4], d[5]] for d in mapped]
            if SAVE_LABELS:
                label_path = cfg.output_dir / f"{base}.txt"
                ensure_dir(label_path.parent)
                from src.full_pipeline.io.save import write_labels

                write_labels(label_path, final_dets)
            if SAVE_COMPOSITES:
                comp_dir = cfg.output_dir / "composites"
                ensure_dir(comp_dir)
                from src.full_pipeline.io.save import save_image

                save_image(comp_dir / f"{base}_composite.jpg", composite)
            t4 = time.perf_counter()

            rows.append(
                TimingRow(
                    image=base,
                    load_ms=(t1 - t0) * 1000,
                    composite_ms=(t2 - t1) * 1000,
                    infer_ms=(t3 - t2) * 1000,
                    post_ms=(t4 - t3) * 1000,
                    total_ms=(t4 - t0) * 1000,
                )
            )

    summary = summarize(rows)

    csv_path = OUTPUT_DIR / "timings.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "load_ms", "composite_ms", "infer_ms", "post_ms", "total_ms"])
        for r in rows:
            writer.writerow(
                [
                    r.image,
                    f"{r.load_ms:.3f}",
                    f"{r.composite_ms:.3f}",
                    f"{r.infer_ms:.3f}",
                    f"{r.post_ms:.3f}",
                    f"{r.total_ms:.3f}",
                ]
            )

    print(f"Wrote per-image timings to: {csv_path}")
    print("Summary (ms):")
    for key, val in summary.items():
        print(f"  {key}: {val:.3f}")


if __name__ == "__main__":
    main()

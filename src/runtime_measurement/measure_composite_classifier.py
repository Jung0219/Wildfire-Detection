"""Measure runtime for the composite + classifier pipeline.

Usage:
    python -m src.runtime_measurement.measure_composite_classifier
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import torch
import yaml
from tqdm import tqdm

from src.full_pipeline.classifier_stage.gating import verify_with_classifier
from src.full_pipeline.config.config import MRClassifierConfig, ensure_dir
from src.full_pipeline.data.loader import base_name as basename_from_path, load_image
from src.full_pipeline.io.save import save_image, write_labels
from src.full_pipeline.models.classifier import load_classifier
from src.full_pipeline.models.detector import load_detector
from src.full_pipeline.postprocess.mapping import map_detections
from src.full_pipeline.postprocess.nms import apply_nms
from src.full_pipeline.preprocess.composite import prepare_image_for_detection

# ================= CONFIG (edit me) =================
CONFIG_PATH = Path("/lab/projects/fire_smoke_awr/src/full_pipeline/run/batch_run.yaml")
OUTPUT_DIR = Path("/lab/projects/fire_smoke_awr/outputs/runtime_measurement/runs/composite_classifier")
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
    det_ms: float
    map_nms_ms: float
    classifier_ms: float
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
    det_vals = [r.det_ms for r in rows]
    total_vals = [r.total_ms for r in rows]
    return {
        "det_mean_ms": sum(det_vals) / len(det_vals) if det_vals else 0.0,
        "det_p50_ms": percentile(det_vals, 50),
        "det_p90_ms": percentile(det_vals, 90),
        "det_p99_ms": percentile(det_vals, 99),
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
    image_dir = Path(cfg_dict.get("image_dir", cfg.image_dir))
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])


def main() -> None:
    print("Runtime measurement: composite + classifier")
    print(f"Config: {CONFIG_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg_dict = load_cfg(CONFIG_PATH)
    cfg = MRClassifierConfig(
        image_dir=Path(cfg_dict.get("image_dir", MRClassifierConfig().image_dir)),
        output_dir=OUTPUT_DIR,
        detector_weights=Path(cfg_dict.get("detector_weights", cfg_dict.get("DetectorWeights", MRClassifierConfig().detector_weights))),
        classifier_weights=Path(cfg_dict.get("classifier_weights", cfg_dict.get("ClassifierWeights", MRClassifierConfig().classifier_weights))),
        intermediate_size=int(cfg_dict.get("intermediate_size", MRClassifierConfig().intermediate_size)),
        nms_iou_thresh=float(cfg_dict.get("nms_iou_thresh", MRClassifierConfig().nms_iou_thresh)),
        conf_low=float(cfg_dict.get("conf_low", MRClassifierConfig().conf_low)),
        conf_high=float(cfg_dict.get("conf_high", MRClassifierConfig().conf_high)),
        classifier_crop_size=int(cfg_dict.get("classifier_crop_size", MRClassifierConfig().classifier_crop_size)),
        anchor_y_frac=float(cfg_dict.get("anchor_y_frac", MRClassifierConfig().anchor_y_frac)),
        save_debug=False,
        save_composites=SAVE_COMPOSITES,
        device=cfg_dict.get("device", DEVICE),
    )
    classifier_type = cfg_dict.get("classifier_type", cfg_dict.get("ClassifierType", "yolo")).lower()

    detector = load_detector(cfg.detector_weights, device=cfg.device)
    classifier = load_classifier(cfg.classifier_weights, model_type=classifier_type, device=cfg.device)
    images = collect_images(cfg_dict, cfg)
    if not images:
        raise ValueError("No images found for measurement.")

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
            t4 = time.perf_counter()

            sync_cuda()
            verified, counts = verify_with_classifier(
                final_dets,
                image,
                classifier,
                cfg.conf_low,
                cfg.conf_high,
                cfg.classifier_crop_size,
            )
            sync_cuda()
            t5 = time.perf_counter()

            if SAVE_LABELS:
                label_path = cfg.output_dir / f"{base}.txt"
                ensure_dir(label_path.parent)
                write_labels(label_path, verified)
            if SAVE_COMPOSITES:
                comp_dir = cfg.output_dir / "composites"
                ensure_dir(comp_dir)
                save_image(comp_dir / f"{base}_composite.jpg", composite)
            t6 = time.perf_counter()

            rows.append(
                TimingRow(
                    image=base,
                    load_ms=(t1 - t0) * 1000,
                    composite_ms=(t2 - t1) * 1000,
                    det_ms=(t3 - t2) * 1000,
                    map_nms_ms=(t4 - t3) * 1000,
                    classifier_ms=(t5 - t4) * 1000,
                    post_ms=(t6 - t5) * 1000,
                    total_ms=(t6 - t0) * 1000,
                )
            )

    summary = summarize(rows)

    csv_path = OUTPUT_DIR / "timings.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image",
                "load_ms",
                "composite_ms",
                "det_ms",
                "map_nms_ms",
                "classifier_ms",
                "post_ms",
                "total_ms",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.image,
                    f"{r.load_ms:.3f}",
                    f"{r.composite_ms:.3f}",
                    f"{r.det_ms:.3f}",
                    f"{r.map_nms_ms:.3f}",
                    f"{r.classifier_ms:.3f}",
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

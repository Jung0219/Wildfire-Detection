"""Single-image multiresolution + two-stage runner (detector + classifier gating).

Edit `CONFIG` to point at your YAML before running:
    python src/full_pipeline/run/single/two_stage_single.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import time
import torch
import yaml

NUM_ITERATIONS = 100  # Number of iterations for runtime measurement

# Ensure repo root on sys.path for direct invocation
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../fire_smoke_awr
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.full_pipeline.classifier_stage.gating import verify_with_classifier
from src.full_pipeline.config.config import MRClassifierConfig, ensure_dir
from src.full_pipeline.data.loader import base_name, load_image
from src.full_pipeline.io.save import save_image, write_labels
from src.full_pipeline.models.classifier import load_classifier
from src.full_pipeline.models.detector import load_detector
from src.full_pipeline.postprocess.mapping import map_detections
from src.full_pipeline.postprocess.nms import apply_nms
from src.full_pipeline.preprocess.composite import prepare_image_for_detection

# ================= CONFIG =================
CONFIG = Path(__file__).resolve().parents[0] / "config.yaml"
# ==========================================


def load_cfg(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def require(cfg: dict, keys: list[str]):
    for key in keys:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    raise ValueError(f"Missing required config value for keys: {keys}")


def main() -> None:
    config_path = Path(CONFIG)
    cfg_dict = load_cfg(config_path)

    cfg = MRClassifierConfig(
        output_dir=Path(require(cfg_dict, ["output_dir"])),
        detector_weights=Path(require(cfg_dict, ["detector_weights", "DetectorWeights"])),
        classifier_weights=Path(require(cfg_dict, ["classifier_weights", "ClassifierWeights"])),
        intermediate_size=int(require(cfg_dict, ["intermediate_size"])),
        nms_iou_thresh=float(cfg_dict.get("nms_iou_thresh", 0.0)),
        conf_low=float(require(cfg_dict, ["conf_low"])),
        conf_high=float(require(cfg_dict, ["conf_high"])),
        classifier_crop_size=int(require(cfg_dict, ["classifier_crop_size"])),
        anchor_y_frac=float(require(cfg_dict, ["anchor_y_frac"])),
        save_debug=bool(cfg_dict.get("save_debug", False)),
        save_composites=bool(cfg_dict.get("save_composites", False)),
        device=cfg_dict.get("device"),
    )

    ensure_dir(cfg.output_dir)
    saved_cfg_path = cfg.output_dir / "config_used.yaml"
    with open(saved_cfg_path, "w") as f:
        yaml.safe_dump(cfg_dict, f)

    classifier_type = cfg_dict.get("classifier_type", cfg_dict.get("ClassifierType", "yolo")).lower()

    detector = load_detector(cfg.detector_weights, device=cfg.device)
    classifier = load_classifier(cfg.classifier_weights, model_type=classifier_type, device=cfg.device)

    image_path = Path(require(cfg_dict, ["image_path", "image"]))
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = load_image(image_path)
    
    # Inference Time Measurement
    # Warm-up round
    for _ in range(10):
        composite, meta = prepare_image_for_detection(image, cfg.intermediate_size, cfg.anchor_y_frac) # preprocessing

        yolo_res = detector.predict(composite, imgsz=640, conf=cfg_dict.get("detector_conf", 0.001), verbose=False)[0]
        mapped = map_detections(yolo_res, meta, image.shape)

        if cfg.nms_iou_thresh and cfg.nms_iou_thresh > 0:
            dets_for_stage = apply_nms(mapped, cfg.nms_iou_thresh)
        else:
            dets_for_stage = [[d[0], d[1], d[2], d[3], d[4], d[5]] for d in mapped]

        verified, counts = verify_with_classifier(
            dets_for_stage, image, classifier, cfg.conf_low, cfg.conf_high, cfg.classifier_crop_size
        )
    
    # Measurement
    run_times = []
    composite_times = []
    map_times = []
    nms_times = []
    verify_times = []
    for _ in range(NUM_ITERATIONS):
        torch.cuda.synchronize()
        start_time = time.time()

        # composite block
        # ===============================================================================================
        composite, meta = prepare_image_for_detection(image, cfg.intermediate_size, cfg.anchor_y_frac) # preprocessing
        torch.cuda.synchronize()
        after_composite = time.time()

        yolo_res = detector.predict(composite, imgsz=640, conf=cfg_dict.get("detector_conf", 0.001), verbose=False)[0]
        mapped = map_detections(yolo_res, meta, image.shape)
        torch.cuda.synchronize()
        after_mapping = time.time()

        if cfg.nms_iou_thresh and cfg.nms_iou_thresh > 0:
            dets_for_stage = apply_nms(mapped, cfg.nms_iou_thresh)
        else:
            dets_for_stage = [[d[0], d[1], d[2], d[3], d[4], d[5]] for d in mapped]
        torch.cuda.synchronize()
        after_nms = time.time()

        verified, counts = verify_with_classifier(
            dets_for_stage, image, classifier, cfg.conf_low, cfg.conf_high, cfg.classifier_crop_size
        )
        torch.cuda.synchronize()
        after_verify = time.time()
        # ===============================================================================================

        run_times.append(after_verify - start_time)
        composite_times.append(after_composite - start_time)
        map_times.append(after_mapping - after_composite)
        nms_times.append(after_nms - after_mapping)
        verify_times.append(after_verify - after_nms)
    
    avg_time = sum(run_times) / len(run_times)
    avg_composite = sum(composite_times) / len(composite_times)
    avg_map = sum(map_times) / len(map_times)
    avg_nms = sum(nms_times) / len(nms_times)
    avg_verify = sum(verify_times) / len(verify_times)
    # save runtime in a file
    runtime_path = cfg.output_dir / "runtime.txt"
    with open(runtime_path, "w") as f:
        for run, runtime in enumerate(run_times):
            f.write(f"Run {run + 1} Inference Time (ms): {runtime * 1000:.3f}\n")
        f.write("\nAverage Breakdown (ms):\n")
        f.write(f"Composite: {avg_composite * 1000:.3f}\n")
        f.write(f"Detect + Map: {avg_map * 1000:.3f}\n")
        f.write(f"NMS: {avg_nms * 1000:.3f}\n")
        f.write(f"Verify: {avg_verify * 1000:.3f}\n")
        f.write(f"Total: {avg_time * 1000:.3f}\n")
    print(
        f"Average Inference Time (ms): {avg_time * 1000:.3f} | "
        f"Composite: {avg_composite * 1000:.3f}, "
        f"Detect + Map: {avg_map * 1000:.3f}, "
        f"NMS: {avg_nms * 1000:.3f}, "
        f"Verify: {avg_verify * 1000:.3f}"
    )


if __name__ == "__main__":
    main()

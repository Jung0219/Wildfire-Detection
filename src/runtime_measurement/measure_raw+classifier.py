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
        yolo_res = detector.predict(image, imgsz=640, conf=cfg_dict.get("detector_conf", 0.001), verbose=False)[0]

        dets_with_xyxy = []
        for box_norm, box_xyxy, conf, cls_id in zip(
            yolo_res.boxes.xywhn.cpu().numpy(),
            yolo_res.boxes.xyxy.cpu().numpy(),
            yolo_res.boxes.conf.cpu().numpy(),
            yolo_res.boxes.cls.cpu().numpy(),
        ):
            dets_with_xyxy.append(
                [
                    int(cls_id),
                    float(box_norm[0]),
                    float(box_norm[1]),
                    float(box_norm[2]),
                    float(box_norm[3]),
                    float(conf),
                    float(box_xyxy[0]),
                    float(box_xyxy[1]),
                    float(box_xyxy[2]),
                    float(box_xyxy[3]),
                ]
            )

        if cfg.nms_iou_thresh and cfg.nms_iou_thresh > 0:
            dets_for_stage = apply_nms(dets_with_xyxy, cfg.nms_iou_thresh)
        else:
            dets_for_stage = [[d[0], d[1], d[2], d[3], d[4], d[5]] for d in dets_with_xyxy]

        verified, counts = verify_with_classifier(
            dets_for_stage, image, classifier, cfg.conf_low, cfg.conf_high, cfg.classifier_crop_size
        )
    
    # Measurement
    run_times = []
    for _ in range(NUM_ITERATIONS):
        torch.cuda.synchronize()
        start_time = time.time()

        # raw + classifier block
        # ===============================================================================================
        yolo_res = detector.predict(image, imgsz=640, conf=cfg_dict.get("detector_conf", 0.001), verbose=False)[0]

        dets_with_xyxy = []
        for box_norm, box_xyxy, conf, cls_id in zip(
            yolo_res.boxes.xywhn.cpu().numpy(),
            yolo_res.boxes.xyxy.cpu().numpy(),
            yolo_res.boxes.conf.cpu().numpy(),
            yolo_res.boxes.cls.cpu().numpy(),
        ):
            dets_with_xyxy.append(
                [
                    int(cls_id),
                    float(box_norm[0]),
                    float(box_norm[1]),
                    float(box_norm[2]),
                    float(box_norm[3]),
                    float(conf),
                    float(box_xyxy[0]),
                    float(box_xyxy[1]),
                    float(box_xyxy[2]),
                    float(box_xyxy[3]),
                ]
            )

        if cfg.nms_iou_thresh and cfg.nms_iou_thresh > 0:
            dets_for_stage = apply_nms(dets_with_xyxy, cfg.nms_iou_thresh)
        else:
            dets_for_stage = [[d[0], d[1], d[2], d[3], d[4], d[5]] for d in dets_with_xyxy]

        verified, counts = verify_with_classifier(
            dets_for_stage, image, classifier, cfg.conf_low, cfg.conf_high, cfg.classifier_crop_size
        )
        # ===============================================================================================

        torch.cuda.synchronize()
        end_time = time.time()
        run_times.append(end_time - start_time) 
    
    avg_time = sum(run_times) / len(run_times)
    # save runtime in a file
    runtime_path = cfg.output_dir / "runtime.txt"
    with open(runtime_path, "w") as f:
        for run, runtime in enumerate(run_times):
            f.write(f"Run {run + 1} Inference Time (ms): {runtime * 1000:.3f}\n")
    print(f"Average Inference Time (ms): {avg_time * 1000:.3f}")


if __name__ == "__main__":
    main()

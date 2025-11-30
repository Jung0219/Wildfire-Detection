"""Single-image multiresolution composite YOLO runner (no classifier).

Edit `CONFIG` to point at your YAML before running:
    python src/full_pipeline/run/single/composite_single.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import yaml

# Ensure repo root on sys.path for direct invocation
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.full_pipeline.config.config import MRClassifierConfig, ensure_dir
from src.full_pipeline.data.loader import base_name, load_image
from src.full_pipeline.io.save import save_image, write_labels
from src.full_pipeline.models.detector import load_detector
from src.full_pipeline.postprocess.mapping import map_detections
from src.full_pipeline.postprocess.nms import apply_nms
from src.full_pipeline.preprocess.composite import prepare_image_for_detection

# ================= CONFIG =================
CONFIG = Path(__file__).resolve().parents[2] / "batch_run.yaml"
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
        intermediate_size=int(require(cfg_dict, ["intermediate_size"])),
        nms_iou_thresh=float(cfg_dict.get("nms_iou_thresh", 0.0)),
        anchor_y_frac=float(require(cfg_dict, ["anchor_y_frac"])),
        save_debug=bool(cfg_dict.get("save_debug", False)),
        save_composites=bool(cfg_dict.get("save_composites", False)),
        device=cfg_dict.get("device"),
    )

    ensure_dir(cfg.output_dir)
    saved_cfg_path = cfg.output_dir / "config_used.yaml"
    with open(saved_cfg_path, "w") as f:
        yaml.safe_dump(cfg_dict, f)

    detector = load_detector(cfg.detector_weights, device=cfg.device)

    image_path = Path(require(cfg_dict, ["image_path", "image"]))
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = load_image(image_path)
    composite, meta = prepare_image_for_detection(image, cfg.intermediate_size, cfg.anchor_y_frac)

    yolo_res = detector.predict(composite, imgsz=640, conf=cfg_dict.get("detector_conf", 0.001), verbose=False)[0]
    mapped = map_detections(yolo_res, meta, image.shape)

    if cfg.nms_iou_thresh and cfg.nms_iou_thresh > 0:
        final_dets = apply_nms(mapped, cfg.nms_iou_thresh)
    else:
        final_dets = [[d[0], d[1], d[2], d[3], d[4], d[5]] for d in mapped]

    labels_path = cfg.output_dir / f"{base_name(image_path)}.txt"
    write_labels(labels_path, final_dets)

    if cfg.save_composites or cfg.save_debug:
        comp_dir = Path(cfg_dict.get("debug_dir", cfg.output_dir / "composites"))
        save_image(comp_dir / f"{base_name(image_path)}_composite.jpg", composite)

    print(f"Processed 1 image: {image_path.name}. Labels at {labels_path}")
    print(f"Config saved to: {saved_cfg_path}")


if __name__ == "__main__":
    main()

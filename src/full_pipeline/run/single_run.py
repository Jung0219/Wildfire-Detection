"""Single-image runner for the modular MR + classifier pipeline."""

from __future__ import annotations

import sys
import yaml
from pathlib import Path

# Ensure repo root on sys.path for direct invocation (python run.py ...)
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.full_pipeline.api import run_mr_classifier_on_image
from src.full_pipeline.config.config import MRClassifierConfig, ensure_dir
from src.full_pipeline.models.classifier import load_classifier
from src.full_pipeline.models.detector import load_detector

# ================= CONFIG =================
CONFIG = "/lab/projects/fire_smoke_awr/src/full_pipeline/run/config.yaml"
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
    # Edit this path or set via environment before running.
    config_path = Path(CONFIG)
    cfg_dict = load_cfg(config_path)

    cfg = MRClassifierConfig(
        image_dir=Path(cfg_dict.get("gt_dir", Path(require(cfg_dict, ["image_dir", "image_path"])).parent.parent)),
        output_dir=Path(require(cfg_dict, ["output_dir"])),
        detector_weights=Path(require(cfg_dict, ["detector_weights", "DetectorWeights"])),
        classifier_weights=Path(require(cfg_dict, ["classifier_weights", "ClassifierWeights"])),
        intermediate_size=int(require(cfg_dict, ["intermediate_size"])),
        nms_iou_thresh=float(require(cfg_dict, ["nms_iou_thresh"])),
        conf_low=float(require(cfg_dict, ["conf_low"])),
        conf_high=float(require(cfg_dict, ["conf_high"])),
        classifier_crop_size=int(require(cfg_dict, ["classifier_crop_size"])),
        anchor_y_frac=float(require(cfg_dict, ["anchor_y_frac"])),
        save_debug=bool(cfg_dict.get("save_debug", False)),
        save_composites=bool(cfg_dict.get("save_composites", False)),
        device=cfg_dict.get("device"),
    )

    classifier_type = cfg_dict.get("classifier_type", cfg_dict.get("ClassifierType", "yolo")).lower()

    detector = load_detector(cfg.detector_weights, device=cfg.device)
    classifier = load_classifier(cfg.classifier_weights, model_type=classifier_type, device=cfg.device)

    image_path = Path(require(cfg_dict, ["image_path", "image"]))
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Persist the exact config alongside outputs
    ensure_dir(cfg.output_dir)
    saved_cfg_path = cfg.output_dir / "args.yaml"
    with open(saved_cfg_path, "w") as f:
        yaml.safe_dump(cfg_dict, f)

    result = run_mr_classifier_on_image(image_path, cfg, detector, classifier)

    print(f"Processed 1 image: {result.base_name}. Labels at {cfg.output_dir}")
    print(f"Config saved to: {saved_cfg_path}")


if __name__ == "__main__":
    main()

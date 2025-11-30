"""YAML-driven runner for the modular MR + classifier pipeline."""

from __future__ import annotations

import sys
import yaml
from pathlib import Path
from tqdm import tqdm

# Ensure repo root on sys.path for direct invocation (python run.py ...)
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.full_pipeline.api import run_mr_classifier_on_image
from src.full_pipeline.config.config import MRClassifierConfig, ensure_dir
from src.full_pipeline.models.classifier import load_classifier
from src.full_pipeline.models.detector import load_detector

# ================= CONFIG =================
CONFIG = "/lab/projects/fire_smoke_awr/src/full_pipeline/run/batch_run.yaml"
# ==========================================

def load_cfg(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    # Edit this path or set via environment before running.
    config_path = Path(CONFIG)
    cfg_dict = load_cfg(config_path)

    cfg = MRClassifierConfig(
        image_dir=Path(cfg_dict.get("image_dir", MRClassifierConfig().image_dir)),
        output_dir=Path(cfg_dict.get("output_dir", MRClassifierConfig().output_dir)),
        detector_weights=Path(
            cfg_dict.get("detector_weights", cfg_dict.get("DetectorWeights", MRClassifierConfig().detector_weights))
        ),
        classifier_weights=Path(
            cfg_dict.get("classifier_weights", cfg_dict.get("ClassifierWeights", MRClassifierConfig().classifier_weights))
        ),
        intermediate_size=int(cfg_dict.get("intermediate_size", MRClassifierConfig().intermediate_size)),
        nms_iou_thresh=float(cfg_dict.get("nms_iou_thresh", MRClassifierConfig().nms_iou_thresh)),
        conf_low=float(cfg_dict.get("conf_low", MRClassifierConfig().conf_low)),
        conf_high=float(cfg_dict.get("conf_high", MRClassifierConfig().conf_high)),
        classifier_crop_size=int(cfg_dict.get("classifier_crop_size", MRClassifierConfig().classifier_crop_size)),
        anchor_y_frac=float(cfg_dict.get("anchor_y_frac", MRClassifierConfig().anchor_y_frac)),
        save_debug=bool(cfg_dict.get("save_debug", MRClassifierConfig().save_debug)),
        save_composites=bool(cfg_dict.get("save_composites", MRClassifierConfig().save_composites)),
        device=cfg_dict.get("device", MRClassifierConfig().device),
    )

    classifier_type = cfg_dict.get("classifier_type", cfg_dict.get("ClassifierType", "yolo")).lower()

    detector = load_detector(cfg.detector_weights, device=cfg.device)
    classifier = load_classifier(cfg.classifier_weights, model_type=classifier_type, device=cfg.device)

    image_dir = Path(cfg_dict.get("image_dir", cfg.image_dir / "images/test"))
    image_files = [f for f in image_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    # Persist the exact config alongside outputs
    ensure_dir(cfg.output_dir)
    saved_cfg_path = cfg.output_dir / "args.yaml"
    with open(saved_cfg_path, "w") as f:
        yaml.safe_dump(cfg_dict, f)

    for img_path in tqdm(sorted(image_files), desc="Running MR+classifier"):
        run_mr_classifier_on_image(img_path, cfg, detector, classifier)

    print(f"Processed {len(image_files)} images. Labels at {cfg.output_dir}")
    print(f"Config saved to: {saved_cfg_path}")


if __name__ == "__main__":
    main()

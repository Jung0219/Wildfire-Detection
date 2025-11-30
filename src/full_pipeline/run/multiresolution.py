"""Run only the multiresolution detector (no classifier gating).

Edits the config path in this script to point at your YAML before running.
"""

from __future__ import annotations

import sys
import yaml
from pathlib import Path
from tqdm import tqdm

# Ensure repo root on sys.path for direct invocation (python multiresolution.py)
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.full_pipeline.config.config import MRClassifierConfig, ensure_dir
from src.full_pipeline.data.loader import base_name as basename_from_path
from src.full_pipeline.data.loader import load_image
from src.full_pipeline.io.save import save_image, write_labels
from src.full_pipeline.models.detector import load_detector
from src.full_pipeline.postprocess.mapping import map_detections
from src.full_pipeline.postprocess.nms import apply_nms
from src.full_pipeline.preprocess.composite import prepare_image_for_detection

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
    config_path = Path(CONFIG)
    cfg_dict = load_cfg(config_path)

    cfg = MRClassifierConfig(
        image_dir=Path(cfg_dict.get("gt_dir", Path(require(cfg_dict, ["image_dir", "image_path"])).parent.parent)),
        output_dir=Path(require(cfg_dict, ["output_dir"])),
        detector_weights=Path(require(cfg_dict, ["detector_weights", "DetectorWeights"])),
        intermediate_size=int(require(cfg_dict, ["intermediate_size"])),
        nms_iou_thresh=float(require(cfg_dict, ["nms_iou_thresh"])),
        anchor_y_frac=float(require(cfg_dict, ["anchor_y_frac"])),
        save_debug=bool(cfg_dict.get("save_debug", False)),
        save_composites=bool(cfg_dict.get("save_composites", False)),
        device=cfg_dict.get("device"),
    )

    detector = load_detector(cfg.detector_weights, device=cfg.device)

    # Build list of images (single or directory)
    image_paths = []
    if cfg_dict.get("image_path"):
        image_paths.append(Path(cfg_dict["image_path"]))
    if cfg_dict.get("image"):
        image_paths.append(Path(cfg_dict["image"]))
    if not image_paths and cfg_dict.get("image_dir"):
        image_dir = Path(cfg_dict["image_dir"])
        image_paths.extend([f for f in image_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    if not image_paths:
        raise ValueError("No images provided. Set image_path or image_dir in config.")

    ensure_dir(cfg.output_dir)
    saved_cfg_path = cfg.output_dir / "config_used.yaml"
    with open(saved_cfg_path, "w") as f:
        yaml.safe_dump(cfg_dict, f)

    for img_path in tqdm(sorted(image_paths), desc="Running multiresolution"):
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = load_image(img_path)
        base = basename_from_path(img_path)

        composite, meta = prepare_image_for_detection(image, cfg.intermediate_size, cfg.anchor_y_frac)

        yolo_res = detector.predict(composite, imgsz=640, conf=0.001, verbose=False)[0]
        mapped = map_detections(yolo_res, meta, image.shape)
        if cfg.nms_iou_thresh and cfg.nms_iou_thresh > 0:
            final_dets = apply_nms(mapped, cfg.nms_iou_thresh)
        else:
            final_dets = [[d[0], d[1], d[2], d[3], d[4], d[5]] for d in mapped]

        labels_path = cfg.output_dir / f"{base}.txt"
        write_labels(labels_path, final_dets)

        if cfg.save_composites or cfg.save_debug:
            comp_dir = cfg.debug_dir if cfg.save_debug else (cfg.output_dir / "composites")
            save_image(comp_dir / f"{base}_composite.jpg", composite)

    print(f"Processed {len(image_paths)} images. Labels at {cfg.output_dir}")
    print(f"Config saved to: {saved_cfg_path}")


if __name__ == "__main__":
    main()

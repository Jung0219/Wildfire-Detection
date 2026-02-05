"""Single-image raw YOLO runner (no composite or classifier).

Edit `CONFIG` to point at your YAML before running:
    python src/full_pipeline/run/single/raw_single.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import yaml
import time
import torch

NUM_ITERATIONS = 100

# Ensure repo root on sys.path for direct invocation
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../fire_smoke_awr
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.full_pipeline.config.config import ensure_dir
from src.full_pipeline.data.loader import base_name, load_image
from src.full_pipeline.io.save import write_labels
from src.full_pipeline.models.detector import load_detector

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


def detections_from_result(yolo_result):
    """Convert YOLO results to det list with xyxy for optional NMS."""

    dets = []
    for box_norm, conf, cls_id, xyxy in zip(
        yolo_result.boxes.xywhn.cpu().numpy(),
        yolo_result.boxes.conf.cpu().numpy(),
        yolo_result.boxes.cls.cpu().numpy(),
        yolo_result.boxes.xyxy.cpu().numpy(),
    ):
        x1, y1, x2, y2 = xyxy.tolist()
        dets.append(
            [
                int(cls_id),
                float(box_norm[0]),
                float(box_norm[1]),
                float(box_norm[2]),
                float(box_norm[3]),
                float(conf),
                float(x1),
                float(y1),
                float(x2),
                float(y2),
            ]
        )
    return dets


def main() -> None:
    config_path = Path(CONFIG)
    cfg_dict = load_cfg(config_path)

    output_dir = Path(require(cfg_dict, ["output_dir"]))
    ensure_dir(output_dir)

    detector_weights = Path(require(cfg_dict, ["detector_weights", "DetectorWeights"]))
    detector = load_detector(detector_weights, device=cfg_dict.get("device"))

    image_path = Path(require(cfg_dict, ["image_path", "image"]))
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = load_image(image_path)
    
    # Inference Time Measurement
    # Warm-up
    for _ in range(10):
        _ = detector.predict(image, imgsz=640, conf=cfg_dict.get("detector_conf", 0.001), verbose=False)[0]
    
    # Measurement
    run_times = []
    for _ in range(NUM_ITERATIONS):
        torch.cuda.synchronize()
        start_time = time.time()
        
        # YOLO Inference
        # ===============================================================================================
        _ = detector.predict(image, imgsz=640, conf=cfg_dict.get("detector_conf", 0.001), verbose=False)[0]
        # ===============================================================================================
        
        torch.cuda.synchronize()
        end_time = time.time()
        run_times.append(end_time - start_time)
        
    # Saving
    avg_run_time = sum(run_times) / len(run_times) if run_times else 0  
    runtime_path = output_dir / "runtime.txt"

    with open(runtime_path, "w") as f:
        for run, runtime in enumerate(run_times):
            f.write(f"Run {run + 1} Inference Time (ms): {runtime * 1000:.3f}\n")
        
    print(f"Average Inference Time (ms): {avg_run_time * 1000:.3f}")

if __name__ == "__main__":
    main()

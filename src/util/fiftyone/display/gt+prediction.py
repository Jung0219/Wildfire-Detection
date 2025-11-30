"""Launch a FiftyOne app to compare YOLO ground truth and multiple prediction sets.

Example:
    # Edit CONFIG below, then run:
    python src/util/fiftyone/display/gt+prediction.py
"""

import os
from pathlib import Path

# ================== CONFIG ==================
# Edit these paths directly to point at your dataset and predictions.
ROOT_DIR = Path(__file__).resolve().parents[4]
DATASET_NAME = "check"

# Base dataset folder (expects images/ and labels/ inside)
PARENT_DIR = Path("/lab/projects/fire_smoke_awr/data/detection/training/pyro-sdis/phash10/original")
IMAGES_SUBDIR = "images/test"  # e.g., "images" or "images/test"
LABELS_SUBDIR = "labels/test"  # e.g., "labels" or "labels/test"
SPLIT = ""  # optional split like "test" or "val"; leave empty to use subdirs above

# Add as many prediction runs as needed; keys become field names in FiftyOne.
PREDICTIONS = {
    "mr_only": Path(
        "/lab/projects/fire_smoke_awr/outputs/yolo/detection/pyro-sdis/phash10/900/es_test/composites"
    ),
    "full_pipeline": Path(
        "/lab/projects/fire_smoke_awr/outputs/yolo/detection/pyro-sdis/phash10/900/es_test/full_pipeline/yolo_0.1_0.3"
    ),
}

# FiftyOne database location (keep writable but inside the repo tree)
DATABASE_DIR = ROOT_DIR / "outputs" / "fiftyone_db"

# --- Port Config ---
USE_ALT_PORT = True
DEFAULT_PORT = 5252
# ============================================

# Force FiftyOne to use the project-local DB before it initializes Mongo
os.environ.setdefault("FIFTYONE_DATABASE_DIR", str(DATABASE_DIR))

import fiftyone as fo
import fiftyone.core.labels as fol
from tqdm import tqdm

IMAGES_DIR = PARENT_DIR / IMAGES_SUBDIR
GT_DIR = PARENT_DIR / LABELS_SUBDIR
if SPLIT:
    IMAGES_DIR = IMAGES_DIR / SPLIT
    GT_DIR = GT_DIR / SPLIT
CLASS_LIST = ["fire", "smoke"]


def validate_path(path: Path, description: str) -> Path:
    """Ensure required directories exist before loading."""
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def load_yolo_labels(images_dir: Path, labels_dir: Path) -> dict[str, fo.Sample]:
    """Generic YOLO loader for GT or predictions (fire/smoke only)."""
    samples_map: dict[str, fo.Sample] = {}
    for fname in tqdm(
        os.listdir(images_dir),
        desc=f"Loading from {labels_dir.name}",
    ):
        if not fname.lower().endswith((".jpg", ".png")):
            continue
        base = os.path.splitext(fname)[0]
        txt_file = labels_dir / f"{base}.txt"
        image_path = images_dir / fname

        if not txt_file.exists() or not image_path.exists():
            continue

        detections = []
        with open(txt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id, x_center, y_center, width, height = map(float, parts[:5])
                confidence = float(parts[5]) if len(parts) == 6 else None
                label = CLASS_LIST[int(class_id)]
                bbox = [x_center - width / 2, y_center - height / 2, width, height]
                det = fol.Detection(label=label, bounding_box=bbox)
                if confidence is not None:
                    det.confidence = confidence
                detections.append(det)

        sample = fo.Sample(filepath=str(image_path))
        sample["temp_field"] = fol.Detections(detections=detections)
        samples_map[str(image_path)] = sample

    return samples_map


def print_config():
    """Emit config so logs capture paths in use."""
    print("Resolved CONFIG:")
    print(f"  ROOT_DIR: {ROOT_DIR}")
    print(f"  PARENT_DIR: {PARENT_DIR}")
    print(f"  IMAGES_SUBDIR: {IMAGES_SUBDIR}")
    print(f"  LABELS_SUBDIR: {LABELS_SUBDIR}")
    print(f"  SPLIT: {SPLIT or '<none>'}")
    print(f"  IMAGES_DIR: {IMAGES_DIR}")
    print(f"  GT_DIR: {GT_DIR}")
    print(f"  DATASET_NAME: {DATASET_NAME}")
    if PREDICTIONS:
        for name, path in PREDICTIONS.items():
            print(f"  PREDICTIONS[{name}]: {path}")
    else:
        print("  PREDICTIONS: <none> (edit the PREDICTIONS dict)")
    print(f"  USE_ALT_PORT: {USE_ALT_PORT}")
    print(f"  DEFAULT_PORT: {DEFAULT_PORT}")
    print(f"  DATABASE_DIR: {DATABASE_DIR}")


def main():
    print_config()

    validate_path(IMAGES_DIR, "Images directory")
    validate_path(GT_DIR, "Ground-truth labels directory")
    for name, path in PREDICTIONS.items():
        validate_path(path, f"Predictions directory '{name}'")

    # --- Configure FiftyOne DB path early to avoid full root volumes ---
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    fo.config.database_dir = str(DATABASE_DIR)
    print(f"FiftyOne database dir set to: {DATABASE_DIR}")

    if DATASET_NAME in fo.list_datasets():
        fo.delete_dataset(DATASET_NAME)
        print("Old dataset deleted.")

    dataset = fo.Dataset(DATASET_NAME, persistent=True)

    # --- Load Ground Truth ---
    gt_map = load_yolo_labels(IMAGES_DIR, GT_DIR)
    samples_dict: dict[str, fo.Sample] = {}
    for path, sample in gt_map.items():
        new_sample = fo.Sample(filepath=path)
        new_sample["ground_truth"] = sample["temp_field"]
        samples_dict[path] = new_sample

    # --- Add extra prediction overlays ---
    for field_name, preds_dir in PREDICTIONS.items():
        preds_map = load_yolo_labels(IMAGES_DIR, preds_dir)
        for path, sample in preds_map.items():
            if path not in samples_dict:
                samples_dict[path] = fo.Sample(filepath=path)
            samples_dict[path][field_name] = sample["temp_field"]

    dataset.add_samples(list(samples_dict.values()))
    print(f"Loaded {len(samples_dict)} samples with fields: {list(PREDICTIONS.keys()) + ['ground_truth']}")

    # --- Launch FiftyOne App ---
    if USE_ALT_PORT:
        try:
            user_port = int(input(f"Enter alternate port (default {DEFAULT_PORT}): ").strip() or DEFAULT_PORT)
        except ValueError:
            print(f"Invalid input. Falling back to default port {DEFAULT_PORT}.")
            user_port = DEFAULT_PORT
        session = fo.launch_app(dataset, port=user_port)
    else:
        session = fo.launch_app(dataset, port=DEFAULT_PORT)

    session.wait()
    print("FiftyOne session closed.")


if __name__ == "__main__":
    main()

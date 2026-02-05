"""Launch a FiftyOne app to visualize YOLO ground-truth labels.

Example:
    # Edit CONFIG below, then run:
    python src/util/fiftyone/display/img+gt.py
"""

import os
from pathlib import Path

# ================== CONFIG ==================
ROOT_DIR = Path(__file__).resolve().parents[4]
DATASET_NAME = "check"

PARENT_DIR = Path("/lab/projects/fire_smoke_awr/data/detection/training/early_smoke/original")
IMAGES_DIR = PARENT_DIR / "images/train"
LABELS_DIR = PARENT_DIR / "labels/train"
CLASS_LIST = ["fire", "smoke"]

# FiftyOne database location (keep writable but inside the repo tree)
DATABASE_DIR = ROOT_DIR / "outputs" / "fiftyone_db"
# ============================================

# Configure DB path before importing FiftyOne to avoid version conflicts
os.environ.setdefault("FIFTYONE_DATABASE_DIR", str(DATABASE_DIR))

import fiftyone as fo
import fiftyone.core.labels as fol
from tqdm import tqdm


def load_yolo_predictions(images_dir: Path, preds_dir: Path, class_list=None):
    samples = []
    for fname in tqdm(os.listdir(images_dir), desc="Loading YOLO Predictions"):
        if not fname.lower().endswith((".jpg", ".png")):
            continue
        base = os.path.splitext(fname)[0]
        txt_file = preds_dir / f"{base}.txt"
        image_path = images_dir / fname

        if not txt_file.exists() or not image_path.exists():
            continue

        detections = []
        with open(txt_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id, x_center, y_center, width, height = map(float, parts[:5])
                confidence = float(parts[5]) if len(parts) == 6 else None
                label = str(int(class_id)) if class_list is None else class_list[int(class_id)]
                bbox = [
                    x_center - width / 2,
                    y_center - height / 2,
                    width,
                    height,
                ]
                det = fol.Detection(label=label, bounding_box=bbox)
                if confidence is not None:
                    det.confidence = confidence
                detections.append(det)

        sample = fo.Sample(filepath=str(image_path), ground_truth=fol.Detections(detections=detections))
        samples.append(sample)

    return samples


def main():
    samples = load_yolo_predictions(IMAGES_DIR, LABELS_DIR, CLASS_LIST)

    if DATASET_NAME in fo.list_datasets():
        fo.delete_dataset(DATASET_NAME)
        print("Old dataset deleted.")

    dataset = fo.Dataset(name=DATASET_NAME, persistent=True)
    dataset.add_samples(samples)

    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    main()

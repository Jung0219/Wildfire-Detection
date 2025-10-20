import os
import json
import xml.etree.ElementTree as ET
import fiftyone as fo
import fiftyone.core.labels as fol
from tqdm import tqdm

# === Set your paths here ===
parent = "/lab/projects/fire_smoke_awr/data/detection/training/early_fire_A_only"
dataset_name = "EF_A_only_test_set"

# === Set your paths here ===
images_dir = f"{parent}/images/test"
anno_path =  f"{parent}/labels/test"  # COCO file or VOC directory or YOLO directory
#anno_path = "/lab/projects/fire_smoke_awr/data/datasets/.HPWREN_FigLib/jsons/hpwren_target_test.json"
anno_type = "yolo"  # "coco", "voc", or "yolo"
class_list = ["fire", "smoke"]  # Example: ["fire", "smoke"] for YOLO, else keep as None

def load_yolo_predictions(images_dir, preds_dir, class_list=None):
    samples = []
    for fname in tqdm(os.listdir(images_dir), desc="Loading YOLO Predictions"):
        if not fname.lower().endswith((".jpg", ".png")):
            continue
        base = os.path.splitext(fname)[0]
        txt_file = os.path.join(preds_dir, base + ".txt")
        image_path = os.path.join(images_dir, fname)

        if not os.path.exists(txt_file) or not os.path.exists(image_path):
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

        sample = fo.Sample(filepath=image_path, ground_truth=fol.Detections(detections=detections))
        samples.append(sample)

    return samples

if __name__ == "__main__":
    samples = load_yolo_predictions(images_dir, anno_path, class_list)

    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
        print("Old dataset deleted.")

    dataset = fo.Dataset(name=dataset_name, persistent=True)
    dataset.add_samples(samples)

    session = fo.launch_app(dataset)
    session.wait()

import os
import fiftyone as fo
import fiftyone.core.labels as fol
from tqdm import tqdm

# ================== CONFIG ==================
parent = "/lab/projects/fire_smoke_awr/data/detection/test_sets/early_fire/dev"
dataset_name = "EF_dev"

extra_predictions = {
    "all" : "/lab/projects/fire_smoke_awr/outputs/yolo/detection/ABCDE_noEF/EF_dev/composites",
    "tp_03": "/lab/projects/fire_smoke_awr/outputs/yolo/detection/ABCDE_noEF/EF_dev/composites/conf_lt_0.3/tp",
    "fp_03":"/lab/projects/fire_smoke_awr/outputs/yolo/detection/ABCDE_noEF/EF_dev/composites/conf_lt_0.3/fp"
} 


# ============================================
images_dir = f"{parent}/images/test"

# Ground truth label folder (YOLO format, classes always fire/smoke)
gt_dir = f"{parent}/labels/test"




CLASS_LIST = ["fire", "smoke"]


def load_yolo_labels(images_dir, labels_dir):
    """Generic YOLO loader for GT or predictions (fire/smoke only)"""
    samples_map = {}
    for fname in tqdm(os.listdir(images_dir), desc=f"Loading from {os.path.basename(labels_dir)}"):
        if not fname.lower().endswith((".jpg", ".png")):
            continue
        base = os.path.splitext(fname)[0]
        txt_file = os.path.join(labels_dir, base + ".txt")
        image_path = os.path.join(images_dir, fname)

        if not os.path.exists(txt_file) or not os.path.exists(image_path):
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

        sample = fo.Sample(filepath=image_path)
        sample["temp_field"] = fol.Detections(detections=detections)
        samples_map[image_path] = sample

    return samples_map


if __name__ == "__main__":
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
        print("Old dataset deleted.")

    dataset = fo.Dataset(dataset_name, persistent=True)

    # --- Load Ground Truth ---
    gt_map = load_yolo_labels(images_dir, gt_dir)
    samples_dict = {}
    for path, sample in gt_map.items():
        new_sample = fo.Sample(filepath=path)
        new_sample["ground_truth"] = sample["temp_field"]
        samples_dict[path] = new_sample

    # --- Add extra prediction overlays ---
    for field_name, preds_dir in extra_predictions.items():
        preds_map = load_yolo_labels(images_dir, preds_dir)
        for path, sample in preds_map.items():
            if path not in samples_dict:
                samples_dict[path] = fo.Sample(filepath=path)
            samples_dict[path][field_name] = sample["temp_field"]

    # Commit to FiftyOne dataset
    dataset.add_samples(list(samples_dict.values()))

    session = fo.launch_app(dataset)
    session.wait()

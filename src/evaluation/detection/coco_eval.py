import os
import json
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ================= CONFIG =================
GT_DIR: str = "/lab/projects/fire_smoke_awr/data/detection/test_sets/early_fire/dev"     # contains images/test and labels/test
PRED_DIR: str = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/ABCDE_noEF/EF_dev/labels" #experiments/nms/nms_0.5"  # e.g., "/path/to/pred_labels" (YOLO txt) 
CONF_THRESH = 0.0  # <--- configurable threshold; None for no filtering
# ==========================================

# Category mapping (adjust if needed)
category_mapping = {
    0: 1,  # fire
    1: 2   # smoke
}
categories = [
    {"id": 1, "name": "fire"},
    {"id": 2, "name": "smoke"}
]

# --- Convert GT labels to COCO ---
def convert_gt_to_coco(gt_dir):
    image_dir = os.path.join(gt_dir, "images", "test")
    label_dir = os.path.join(gt_dir, "labels", "test")
    output_json = os.path.join(gt_dir, "ground_truth.json")

    images, annotations = [], []
    image_id, ann_id = 1, 1

    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".txt")])
    for label_file in tqdm(label_files, desc="Converting GT"):
        img_name = label_file.replace(".txt", ".jpg")
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            print(f"[WARN] Image not found for GT label file: {label_file}")
            continue

        with Image.open(img_path) as img:
            w, h = img.size

        images.append({"id": image_id, "file_name": img_name, "width": w, "height": h})

        with open(os.path.join(label_dir, label_file)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, xc, yc, bw, bh = map(float, parts)
                cls = int(cls)
                if cls not in category_mapping:
                    continue
                bbox_w, bbox_h = bw * w, bh * h
                bbox_x, bbox_y = xc * w - bbox_w / 2, yc * h - bbox_h / 2
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_mapping[cls],
                    "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
                    "area": bbox_w * bbox_h,
                    "iscrowd": 0
                })
                ann_id += 1
        image_id += 1

    coco = {
        "info": {"description": "Converted YOLO dataset", "version": "1.0", "year": 2025},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

    return output_json


# --- Convert Pred labels to COCO ---
def convert_preds_to_coco(gt_json, pred_dir, conf_thresh=0.0):
    output_json = os.path.join(pred_dir, "predictions.json")

    with open(gt_json) as f:
        gt = json.load(f)
    file_to_id = {img["file_name"]: img["id"] for img in gt["images"]}

    detections = []
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".txt")])

    for pred_file in tqdm(pred_files, desc="Converting Predictions"):
        img_name = pred_file.replace(".txt", ".jpg")
        if img_name not in file_to_id:
            print(f"[WARN] No matching GT image for prediction file: {pred_file}")
            continue
        image_id = file_to_id[img_name]

        img_entry = next(x for x in gt["images"] if x["file_name"] == img_name)
        w, h = img_entry["width"], img_entry["height"]

        with open(os.path.join(pred_dir, pred_file)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 6:
                    continue
                cls, xc, yc, bw, bh, conf = map(float, parts)
                if conf_thresh is not None and conf < conf_thresh:   # <--- filter here
                    continue
                cls = int(cls)
                if cls not in category_mapping:
                    continue
                bbox_w, bbox_h = bw * w, bh * h
                bbox_x, bbox_y = xc * w - bbox_w / 2, yc * h - bbox_h / 2
                detections.append({
                    "image_id": image_id,
                    "category_id": category_mapping[cls],
                    "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
                    "score": float(conf)
                })

    with open(output_json, "w") as f:
        json.dump(detections, f, indent=2)

    return output_json


# --- Run evaluation ---
if __name__ == "__main__":
    gt_json = convert_gt_to_coco(GT_DIR)
    pred_json = convert_preds_to_coco(gt_json, PRED_DIR, conf_thresh=CONF_THRESH)

    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(pred_json)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print(PRED_DIR)

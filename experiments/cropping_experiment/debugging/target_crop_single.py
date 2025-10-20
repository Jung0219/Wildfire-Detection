import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os

# ================= CONFIG =================
IMG_PATH     = "/lab/projects/fire_smoke_awr/data/detection/training/early_fire/images/test/AoF05096.jpg"
LABEL_PATH   = "/lab/projects/fire_smoke_awr/data/detection/training/early_fire/labels/test/AoF05096.txt"
YOLO_MODEL   = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/early_fire_pad_aug/train/weights/best.pt"

INTERMEDIATE_SIZE = 800
SAVE_IMG = True
ANCHOR_FROM_GT = True
OUTPUT_DIR = "/lab/projects/fire_smoke_awr/src/data_manipulation/cropping/debugging"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ==========================================


def load_gt_center(label_path, use_mean=False):
    """Return (xc, yc) of first or mean GT box in YOLO format."""
    if not os.path.exists(label_path):
        return (0.5, 0.5)

    with open(label_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if not lines:
        return (0.5, 0.5)

    boxes = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 5:
            xc, yc, w, h = map(float, parts[1:5])
            boxes.append((xc, yc))

    if not boxes:
        return (0.5, 0.5)

    if use_mean:
        arr = np.array(boxes)
        return float(arr[:, 0].mean()), float(arr[:, 1].mean())
    else:
        return boxes[0]


def generate_crop_640x640(original_image, object_center_norm, intermediate_size):
    """Resize → crop dynamically around object center."""
    TARGET_SIZE = 640
    orig_h, orig_w = original_image.shape[:2]

    # 1. Resize so longest side = intermediate_size
    scale_inter = intermediate_size / max(orig_w, orig_h)
    res_inter_w, res_inter_h = int(orig_w * scale_inter), int(orig_h * scale_inter)
    image_inter = cv2.resize(original_image, (res_inter_w, res_inter_h))
    
    print("intermediate shape:", res_inter_h, res_inter_w)

    # 2. Determine 640×640 letterbox remainder
    scale_to_640 = min(TARGET_SIZE / res_inter_w, TARGET_SIZE / res_inter_h)
    resized_w = int(res_inter_w * scale_to_640)
    resized_h = int(res_inter_h * scale_to_640)

    print("resized shape:", resized_h, resized_w)

    crop_h = TARGET_SIZE - resized_h
    crop_w = resized_w

    print("window shape: ", crop_h, crop_w)

    # 3. Object center (in intermediate coordinates)
    obj_x = int(np.clip(object_center_norm[0], 0, 1) * res_inter_w)
    obj_y = int(np.clip(object_center_norm[1], 0, 1) * res_inter_h)

    # 4. Define crop
    crop_x1 = int(obj_x - crop_w / 2)
    crop_y1 = int(obj_y - crop_h / 2)
    crop_x2 = int(obj_x + crop_w / 2)
    crop_y2 = int(obj_y + crop_h / 2)
    
    print("initial crop coords (intermediate):", crop_x1, crop_y1, crop_x2, crop_y2)

    # 5. Clamp
    if crop_x1 < 0:
        crop_x2 += -crop_x1
        crop_x1 = 0
    if crop_y1 < 0:
        crop_y2 += -crop_y1
        crop_y1 = 0
    if crop_x2 > res_inter_w:
        crop_x1 -= (crop_x2 - res_inter_w)
        crop_x2 = res_inter_w
    if crop_y2 > res_inter_h:
        crop_y1 -= (crop_y2 - res_inter_h)
        crop_y2 = res_inter_h
    
    if abs(crop_y2 - crop_y1) > res_inter_h:
        crop_y1 = 0
        crop_y2 = res_inter_h
    if abs(crop_x2 - crop_x1) > res_inter_w:
        crop_x1 = 0
        crop_x2 = res_inter_w
        
    print("crop coords (intermediate):", crop_x1, crop_y1, crop_x2, crop_y2)

    cropped = image_inter[crop_y1:crop_y2, crop_x1:crop_x2]
    meta = dict(
        crop_x1=crop_x1, crop_y1=crop_y1, crop_w=crop_w, crop_h=crop_h,
        scale_inter=scale_inter, res_inter_w=res_inter_w, res_inter_h=res_inter_h,
        orig_w=orig_w, orig_h=orig_h
    )
    return cropped, meta


def yolo_to_original_crop_xyxy(box_xyxy, meta, conf, cls_id):
    """Map detection from crop → original coordinates."""
    x1, y1, x2, y2 = map(float, box_xyxy)
    x1 += meta["crop_x1"]; x2 += meta["crop_x1"]
    y1 += meta["crop_y1"]; y2 += meta["crop_y1"]
    x1 /= meta["scale_inter"]; x2 /= meta["scale_inter"]
    y1 /= meta["scale_inter"]; y2 /= meta["scale_inter"]

    ow, oh = meta["orig_w"], meta["orig_h"]
    return [
        int(cls_id),
        ((x1 + x2) / 2) / ow,  # xc
        ((y1 + y2) / 2) / oh,  # yc
        (x2 - x1) / ow,        # w
        (y2 - y1) / oh,        # h
        float(conf)
    ]


if __name__ == "__main__":
    model = YOLO(YOLO_MODEL)

    # Load image
    original = cv2.imread(IMG_PATH)
    H, W = original.shape[:2]

    # Load GT center or fallback
    gt_center = load_gt_center(LABEL_PATH)
    print(f"GT center (normalized): {gt_center}")

    # Generate crop
    cropped, meta = generate_crop_640x640(original, gt_center, INTERMEDIATE_SIZE)
    print(f"Crop region: {meta}")

    # Run inference
    results = model.predict(cropped, imgsz=640, conf=0.001, verbose=False)[0]
    dets = []
    for box_xyxy, conf, cls_id in zip(
        results.boxes.xyxy.cpu().numpy(),
        results.boxes.conf.cpu().numpy(),
        results.boxes.cls.cpu().numpy()
    ):
        mapped = yolo_to_original_crop_xyxy(box_xyxy, meta, conf, cls_id)
        dets.append(mapped)

    print(f"Detections mapped to original scale: {dets}")

    # Visualization
    if SAVE_IMG:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_crop = os.path.join(OUTPUT_DIR, "cropped.jpg")
        out_vis = os.path.join(OUTPUT_DIR, "visualization.jpg")

        cv2.imwrite(out_crop, cropped)

        vis = cv2.resize(original, (meta["res_inter_w"], meta["res_inter_h"]))
        cv2.rectangle(
            vis,
            (meta["crop_x1"], meta["crop_y1"]),
            (meta["crop_x1"] + cropped.shape[1], meta["crop_y1"] + cropped.shape[0]),
            (255, 0, 0),
            2,
        )
        cv2.putText(vis, f"Crop Region ({cropped.shape[1]}x{cropped.shape[0]})",
                    (meta["crop_x1"], max(30, meta["crop_y1"] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imwrite(out_vis, vis)

        print(f"Saved cropped image to: {out_crop}")
        print(f"Saved visualization to: {out_vis}")

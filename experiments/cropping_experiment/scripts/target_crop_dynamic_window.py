import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO, RTDETR
from torchvision.ops import nms
from tqdm import tqdm

# ================= CONFIG =================
GT_DIR      = "/lab/projects/fire_smoke_awr/data/detection/test_sets/early_smoke"
PARENT_DIR  = "/lab/projects/fire_smoke_awr/outputs/RTDETR/early_fire_crop_aug/early_smoke/target_cropping"
MODEL       = "/lab/projects/fire_smoke_awr/outputs/RTDETR/early_fire_crop_aug/train/weights/best.pt"

INTERMEDIATE_SIZE = 800
NMS_IOU_THRESH = None
SAVE_IMG = True
ANCHOR_FROM_GT = True  # <-- Use GT box center instead of skyline
MODE = "RTDETR"
# ==========================================
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--intermediate_size", type=int, default=800)
args = parser.parse_args()
INTERMEDIATE_SIZE = args.intermediate_size
# ==========================================
IMAGE_DIR = os.path.join(GT_DIR, "images/test")
LABEL_DIR = os.path.join(GT_DIR, "labels/test")
OUTPUT_DIR = os.path.join(PARENT_DIR, f"window_{INTERMEDIATE_SIZE}")
cropped_dir = os.path.join(OUTPUT_DIR, "cropped_images")
if SAVE_IMG:
    os.makedirs(cropped_dir, exist_ok=True)
# ==========================================


def load_gt_center(label_path, use_mean=False):
    """Reads YOLO label file and returns (xc, yc) of first or mean box."""
    if not os.path.exists(label_path):
        return None
    with open(label_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if not lines:
        return None

    boxes = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 5:
            xc, yc, w, h = map(float, parts[1:5])
            boxes.append((xc, yc))
    if not boxes:
        return None

    if use_mean:
        arr = np.array(boxes)
        return float(arr[:, 0].mean()), float(arr[:, 1].mean())
    else:
        return boxes[0]


def generate_crop_640x640(original_image, object_center_norm, intermediate_size):
    """Generate dynamic crop window centered on GT."""
    TARGET_SIZE = 640
    orig_h, orig_w = original_image.shape[:2]

    if orig_h > orig_w:
        return original_image, {
            "crop_x1": 0, "crop_y1": 0,
            "crop_w": orig_w, "crop_h": orig_h,
            "scale_inter": 1.0,
            "res_inter_w": orig_w, "res_inter_h": orig_h,
            "orig_w": orig_w, "orig_h": orig_h,
            "note": "Returned original because height > width"
        }

    scale_inter = intermediate_size / max(orig_w, orig_h)
    res_inter_w, res_inter_h = int(orig_w * scale_inter), int(orig_h * scale_inter)
    image_inter = cv2.resize(original_image, (res_inter_w, res_inter_h))

    scale_to_640 = min(TARGET_SIZE / res_inter_w, TARGET_SIZE / res_inter_h)
    resized_w = int(res_inter_w * scale_to_640)
    resized_h = int(res_inter_h * scale_to_640)

    crop_h = TARGET_SIZE - resized_h
    crop_w = resized_w

    obj_x = int(np.clip(object_center_norm[0], 0, 1) * res_inter_w)
    obj_y = int(np.clip(object_center_norm[1], 0, 1) * res_inter_h)

    crop_x1 = int(obj_x - crop_w / 2)
    crop_y1 = int(obj_y - crop_h / 2)
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h

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
        crop_y1 = 0; crop_y2 = res_inter_h
    if abs(crop_x2 - crop_x1) > res_inter_w:
        crop_x1 = 0; crop_x2 = res_inter_w

    cropped = image_inter[crop_y1:crop_y2, crop_x1:crop_x2]
    if cropped.size == 0:
        cropped = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)

    meta = {
        "crop_x1": crop_x1, "crop_y1": crop_y1,
        "crop_w": crop_w, "crop_h": crop_h,
        "scale_inter": scale_inter,
        "res_inter_w": res_inter_w, "res_inter_h": res_inter_h,
        "orig_w": orig_w, "orig_h": orig_h,
    }
    return cropped, meta


def yolo_to_original_crop_xyxy(box_xyxy, meta, conf, cls_id):
    """Map YOLO detection box from crop → original image coordinates."""
    x1, y1, x2, y2 = map(float, box_xyxy)
    x1 += meta["crop_x1"]; x2 += meta["crop_x1"]
    y1 += meta["crop_y1"]; y2 += meta["crop_y1"]
    x1 /= meta["scale_inter"]; x2 /= meta["scale_inter"]
    y1 /= meta["scale_inter"]; y2 /= meta["scale_inter"]
    ow, oh = meta["orig_w"], meta["orig_h"]

    return [
        int(cls_id),
        ((x1 + x2) / 2) / ow,
        ((y1 + y2) / 2) / oh,
        (x2 - x1) / ow,
        (y2 - y1) / oh,
        float(conf),
        x1, y1, x2, y2
    ]


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if MODE == "YOLO":
        model = YOLO(MODEL)
    
    if MODE == "RTDETR":
        model = RTDETR(MODEL)
        
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for img_name in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(IMAGE_DIR, img_name)
        base, ext = os.path.splitext(img_name)

        out_txt = os.path.join(OUTPUT_DIR, base + ".txt")
        out_crop = os.path.join(cropped_dir, base + "_crop.jpg")

        original = cv2.imread(img_path)
        H, W = original.shape[:2]

        # --- Determine object center ---
        label_path = os.path.join(LABEL_DIR, base + ".txt")
        if ANCHOR_FROM_GT:
            gt_center = load_gt_center(label_path)
            obj_center = gt_center if gt_center is not None else (0.5, 0.5)
        else:
            obj_center = (0.5, 0.5)

        # --- Generate crop ---
        cropped, meta = generate_crop_640x640(original, obj_center, INTERMEDIATE_SIZE)

        # --- Run YOLO ---
        results = model.predict(cropped, imgsz=640, conf=0.001, verbose=False)[0]
        dets = []
        mapped_preds = []  # <--- store mapped predictions for drawing
        for box_xyxy, conf, cls_id in zip(
            results.boxes.xyxy.cpu().numpy(),
            results.boxes.conf.cpu().numpy(),
            results.boxes.cls.cpu().numpy()
        ):
            mapped = yolo_to_original_crop_xyxy(box_xyxy, meta, conf, cls_id)
            dets.append(mapped)
            mapped_preds.append(mapped)  # <--- save for visualization

        # --- Visualization ---
        if SAVE_IMG:
            vis_image = cv2.resize(original, (meta["res_inter_w"], meta["res_inter_h"]))

            # Crop boundary (blue)
            crop_x1, crop_y1 = meta["crop_x1"], meta["crop_y1"]
            crop_x2 = crop_x1 + cropped.shape[1]
            crop_y2 = crop_y1 + cropped.shape[0]
            cv2.rectangle(vis_image, (crop_x1, crop_y1), (crop_x2, crop_y2), (255, 0, 0), 1)
            cv2.putText(vis_image, f"CROP ({cropped.shape[1]}x{cropped.shape[0]})",
                        (crop_x1, max(20, crop_y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

            # GT boxes (green)
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id, xc, yc, w, h = map(float, parts[:5])
                            x1 = int((xc - w / 2) * meta["res_inter_w"])
                            y1 = int((yc - h / 2) * meta["res_inter_h"])
                            x2 = int((xc + w / 2) * meta["res_inter_w"])
                            y2 = int((yc + h / 2) * meta["res_inter_h"])
                            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                            cv2.putText(vis_image, f"GT {int(cls_id)}", (x1, max(20, y1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Predictions (red)
            for pred in mapped_preds:
                _, xc, yc, w, h, conf, x1, y1, x2, y2 = pred
                x1, y1 = int(x1 * meta["scale_inter"]), int(y1 * meta["scale_inter"])
                x2, y2 = int(x2 * meta["scale_inter"]), int(y2 * meta["scale_inter"])
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.putText(vis_image, f"Pred {conf:.2f}", (x1, y2 + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imwrite(out_crop, vis_image)

        # --- Save results ---
        with open(out_txt, "w") as f:
            for d in dets:
                f.write(f"{d[0]} {d[1]:.6f} {d[2]:.6f} {d[3]:.6f} {d[4]:.6f} {d[5]:.4f}\n")

    print(f"✅ Finished processing {len(image_files)} images. Results saved to {OUTPUT_DIR}")

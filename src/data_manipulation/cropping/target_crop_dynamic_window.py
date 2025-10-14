import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO
from torchvision.ops import nms
from tqdm import tqdm

# ================= CONFIG =================
GT_DIR      = "/lab/projects/fire_smoke_awr/data/detection/training/early_fire"
PARENT_DIR  = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/early_fire_pad_aug/test_set"
YOLO_MODEL  = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/early_fire_pad_aug/train/weights/best.pt"

INTERMEDIATE_SIZE = 780
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--intermediate_size", type=int, default=780)
args = parser.parse_args()
INTERMEDIATE_SIZE = args.intermediate_size

NMS_IOU_THRESH = None
SAVE_IMG = True
ANCHOR_FROM_GT = True  # <-- Use GT box center instead of skyline
ANCHOR_USE_MEAN = False  # if multiple objects: True=average center, False=first box

# ==========================================
IMAGE_DIR = os.path.join(GT_DIR, "images/test")
LABEL_DIR = os.path.join(GT_DIR, "labels/test")
OUTPUT_DIR = os.path.join(PARENT_DIR, "target_crop_fixed_window")
cropped_dir = os.path.join(OUTPUT_DIR, "cropped_images")
if SAVE_IMG:
    os.makedirs(cropped_dir, exist_ok=True)
# ==========================================


def load_gt_center(label_path, use_mean=False):
    """
    Reads a YOLO-format label file and returns (xc, yc) of the first or mean box.
    """
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


def generate_crop_fixed(original_image, object_center_norm, intermediate_size,
                        crop_w=640, crop_h=280, anchor_x_frac=0.5, anchor_y_frac=0.25):
    """
    Generates a fixed-size crop (default 640x280) around an anchor position 
    derived from the normalized object center. Uses an intermediate upscale.
    """
    orig_h, orig_w = original_image.shape[:2]

    # Step 1: Resize original image so that the longer side = intermediate_size
    scale_inter = intermediate_size / (orig_w if orig_w >= orig_h else orig_h)
    res_inter_w, res_inter_h = int(orig_w * scale_inter), int(orig_h * scale_inter)
    image_inter = cv2.resize(original_image, (res_inter_w, res_inter_h))

    # Step 2: Convert normalized object center into pixel coordinates (intermediate scale)
    obj_x = int(np.clip(object_center_norm[0], 0, 1) * res_inter_w)
    obj_y = int(np.clip(object_center_norm[1], 0, 1) * res_inter_h)

    # Step 3: Compute crop window centered around anchor fractions
    anchor_x = int(round(anchor_x_frac * crop_w))
    anchor_y = int(round(anchor_y_frac * crop_h))

    crop_x1 = max(0, obj_x - anchor_x)
    crop_y1 = max(0, obj_y - anchor_y)
    crop_x2 = min(crop_x1 + crop_w, res_inter_w)
    crop_y2 = min(crop_y1 + crop_h, res_inter_h)

    # Adjust if crop goes out of bounds
    if crop_x2 - crop_x1 < crop_w:
        crop_x1 = max(0, crop_x2 - crop_w)
    if crop_y2 - crop_y1 < crop_h:
        crop_y1 = max(0, crop_y2 - crop_h)

    cropped = image_inter[crop_y1:crop_y2, crop_x1:crop_x2]
    if cropped.size == 0:
        cropped = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)

    meta = {
        "crop_x1": crop_x1, "crop_y1": crop_y1,
        "scale_inter": scale_inter,
        "res_inter_w": res_inter_w, "res_inter_h": res_inter_h,
        "orig_w": orig_w, "orig_h": orig_h
    }
    return cropped, meta



def yolo_to_original_crop_xyxy(box_xyxy, meta, conf, cls_id):
    x1, y1, x2, y2 = map(float, box_xyxy)

    # Map crop → intermediate coordinates
    x1 += meta["crop_x1"]; x2 += meta["crop_x1"]
    y1 += meta["crop_y1"]; y2 += meta["crop_y1"]

    # Map intermediate → original coordinates
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


def apply_nms(dets, iou_thresh, orig_w, orig_h):
    if not dets:
        return []

    boxes = torch.tensor([d[6:10] for d in dets], dtype=torch.float32)  # x1,y1,x2,y2
    scores = torch.tensor([d[5] for d in dets], dtype=torch.float32)
    classes = torch.tensor([d[0] for d in dets], dtype=torch.int64)

    keep = []
    for cls in torch.unique(classes):
        idxs = torch.where(classes == cls)[0]
        keep_idx = nms(boxes[idxs], scores[idxs], iou_thresh)
        keep.extend(idxs[keep_idx].tolist())

    final_dets = []
    for i in keep:
        cls_id, xc, yc, w, h, conf, *_ = dets[i]
        final_dets.append([cls_id, xc, yc, w, h, conf])

    return final_dets


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = YOLO(YOLO_MODEL)
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
            gt_center = load_gt_center(label_path, use_mean=ANCHOR_USE_MEAN)
            if gt_center is not None:
                obj_center = gt_center
            else:
                obj_center = (0.5, 0.5)  # fallback
        else:
            obj_center = (0.5, 0.5)  # fallback if no GT mode

        # --- Generate crop ---
        cropped, meta = generate_crop_fixed(original, obj_center, INTERMEDIATE_SIZE)

        if SAVE_IMG:
            cv2.imwrite(out_crop, cropped)

        # --- Run YOLO ---
        results = model.predict(cropped, imgsz=640, conf=0.001, verbose=False)[0]
        dets = []
        for box_xyxy, conf, cls_id in zip(
            results.boxes.xyxy.cpu().numpy(),
            results.boxes.conf.cpu().numpy(),
            results.boxes.cls.cpu().numpy()
        ):
            mapped = yolo_to_original_crop_xyxy(box_xyxy, meta, conf, cls_id)
            dets.append(mapped)

        # --- Apply NMS if needed ---
        if NMS_IOU_THRESH and NMS_IOU_THRESH > 0:
            final_dets = apply_nms(dets, NMS_IOU_THRESH, W, H)
        else:
            final_dets = [[d[0], d[1], d[2], d[3], d[4], d[5]] for d in dets]

        # --- Save results ---
        with open(out_txt, "w") as f:
            for cls_id, xc, yc, w, h, conf in final_dets:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.4f}\n")

    print(f"✅ Finished processing {len(image_files)} images. Results saved to {OUTPUT_DIR}")

I want to run this with diffeernt intermediate sizes. Give me a shell script that iterates through
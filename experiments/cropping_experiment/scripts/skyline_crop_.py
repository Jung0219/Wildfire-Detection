import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO
from torchvision.ops import nms
from tqdm import tqdm

# ================= CONFIG =================
GT_DIR      = "/lab/projects/fire_smoke_awr/data/detection/training/early_fire"
PARENT_DIR  = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/early_fire_pad_aug/test_set/skyline_crop"
YOLO_MODEL  = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/early_fire_pad_aug/train/weights/best.pt"

INTERMEDIATE_SIZE = 800
NMS_IOU_THRESH = None
SAVE_IMG = True
ANCHOR_FROM_GT = True  # <-- Use GT box center instead of skyline
# ==========================================
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--intermediate_size", type=int, default=780)
args = parser.parse_args()
INTERMEDIATE_SIZE = args.intermediate_size
"""
# ==========================================

IMAGE_DIR = os.path.join(GT_DIR, "images/test")
LABEL_DIR = os.path.join(GT_DIR, "labels/test")
OUTPUT_DIR = os.path.join(PARENT_DIR, f"skyline_crop_window_{INTERMEDIATE_SIZE}")
cropped_dir = os.path.join(OUTPUT_DIR, "cropped_images")
if SAVE_IMG:
    os.makedirs(cropped_dir, exist_ok=True)
# ==========================================

def detect_skyline_y(img_bgr, cb_min=120, cb_max=255, cr_min=0, cr_max=130, sky_ratio_thresh=5.0):
    H, W = img_bgr.shape[:2]
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    y_thresh = float(Y.astype(np.float32).mean())

    def sky_mask(bgr):
        ycrcb_ = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        Y_, Cr_, Cb_ = cv2.split(ycrcb_)
        return ((Y_ >= y_thresh) &
                (Cb_ >= cb_min) & (Cb_ <= cb_max) &
                (Cr_ >= cr_min) & (Cr_ <= cr_max)).astype(np.uint8)

    m_full = sky_mask(img_bgr)
    counts = m_full.sum(axis=1) / float(W)
    d = np.diff(counts)
    idx = int(np.argmin(d))
    y_candidate = int(np.clip(idx + 1, 0, H - 1))
    above = int(m_full[:y_candidate, :].sum())
    below = int(m_full[y_candidate:, :].sum())
    ratio = (above + 1e-9) / (below + 1e-9)
    return y_candidate if ratio >= sky_ratio_thresh else -1


def generate_crop_640x640(original_image, object_center_norm, intermediate_size):
    """
    Resize the image (longest side = intermediate_size).
    Compute the 640-letterbox remainder as crop window size.
    Crop that window around the GT center so that the object
    is as centered as possible within the window.
    """
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

    # === 1. Resize so longest side = intermediate_size ===
    scale_inter = intermediate_size / max(orig_w, orig_h)
    res_inter_w, res_inter_h = int(orig_w * scale_inter), int(orig_h * scale_inter)
    image_inter = cv2.resize(original_image, (res_inter_w, res_inter_h))

    # === 2. Determine what the 640×640 letterbox would look like ===
    scale_to_640 = min(TARGET_SIZE / res_inter_w, TARGET_SIZE / res_inter_h)
    resized_w = int(res_inter_w * scale_to_640)
    resized_h = int(res_inter_h * scale_to_640)

    # The "empty region" defines the dynamic crop window size
    crop_h = TARGET_SIZE - resized_h
    crop_w = resized_w

    # === 3. Compute object center in intermediate coordinates ===
    obj_x = int(np.clip(object_center_norm[0], 0, 1) * res_inter_w)
    obj_y = int(np.clip(object_center_norm[1], 0, 1) * res_inter_h)

    # === 4. Define crop coordinates (centered on the object) ===
    crop_x1 = int(obj_x - crop_w / 2)
    crop_y1 = int(obj_y - crop_h / 2)
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h
    
    # --- 3) Convert skyline (passed in as object_center_norm[1]) to intermediate coords
    # object_center_norm[1] = skyline_y / orig_h (from caller)
    sky_i = int(round(np.clip(object_center_norm[1], 0.0, 1.0) * res_inter_h))

    # X anchor: still center on provided x (default 0.5)
    obj_x = int(round(np.clip(object_center_norm[0], 0.0, 1.0) * res_inter_w))

    # --- 4) Place crop: TOP aligned to skyline (strip below skyline)
    # Clamp ORIGIN so width/height NEVER change
    crop_x1 = int(np.clip(obj_x - crop_w // 2, 0, max(0, res_inter_w - crop_w)))
    crop_y1 = int(np.clip(sky_i,                0, max(0, res_inter_h - crop_h)))
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h    

    # === 6. Crop the intermediate image ===
    cropped = image_inter[crop_y1:crop_y2, crop_x1:crop_x2]
    if cropped.size == 0:
        cropped = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)

    # === 7. Return with metadata ===
    meta = {
        "crop_x1": crop_x1, "crop_y1": crop_y1,
        "crop_w": crop_w, "crop_h": crop_h,
        "scale_inter": scale_inter,
        "res_inter_w": res_inter_w, "res_inter_h": res_inter_h,
        "orig_w": orig_w, "orig_h": orig_h,
    }

    return cropped, meta


def yolo_to_original_crop_xyxy(box_xyxy, meta, conf, cls_id):
    # box_xyxy is in (x1, y1, x2, y2) format relative to the CROP image
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
        ((x1 + x2) / 2) / ow, # xc
        ((y1 + y2) / 2) / oh, # yc
        (x2 - x1) / ow,       # w
        (y2 - y1) / oh,       # h
        float(conf),
        x1, y1, x2, y2
    ]


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
        sky_y = detect_skyline_y(original)
        if sky_y != -1:
            obj_center = (0.5, sky_y / H)  # x fixed at 0.5
        else:
            obj_center = (0.5, 0.25)  # fallback if skyline not detected


        # --- Generate crop ---
        cropped, meta = generate_crop_640x640(original, obj_center, INTERMEDIATE_SIZE)


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

        if SAVE_IMG:
            vis_image = cv2.resize(original, (meta["res_inter_w"], meta["res_inter_h"]))

            # --- Draw crop boundary on intermediate image ---
            crop_x1, crop_y1 = meta["crop_x1"], meta["crop_y1"]
            crop_x2 = crop_x1 + cropped.shape[1]
            crop_y2 = crop_y1 + cropped.shape[0]
            
            # draw skyline
            if sky_y != -1:
                cv2.line(vis_image, (0, int(sky_y * meta["res_inter_h"] / H)), 
                        (meta["res_inter_w"], int(sky_y * meta["res_inter_h"] / H)), (0, 255, 255), 1)
                cv2.putText(vis_image, "Skyline", (5, int(sky_y * meta["res_inter_h"] / H) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)


            cv2.rectangle(vis_image, (crop_x1, crop_y1), (crop_x2, crop_y2), (255, 0, 0), 1)
            cv2.putText(
                vis_image,
                f"CROP REGION ({cropped.shape[1]}x{cropped.shape[0]})",
                (crop_x1, max(20, crop_y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                1,
            )

            # --- Overlay GT boxes (scaled to intermediate size) ---
            label_path = os.path.join(LABEL_DIR, base + ".txt")
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
                            cv2.putText(
                                vis_image,
                                f"GT {int(cls_id)}",
                                (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                1,
                            )

            # --- Save visualization ---
            cv2.imwrite(out_crop, vis_image)

        # --- Save results ---
        with open(out_txt, "w") as f:
            for d in dets:
                f.write(f"{d[0]} {d[1]:.6f} {d[2]:.6f} {d[3]:.6f} {d[4]:.6f} {d[5]:.4f}\n")

    print(f"✅ Finished processing {len(image_files)} images. Results saved to {OUTPUT_DIR}")
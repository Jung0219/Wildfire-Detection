import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from ultralytics.utils.ops import non_max_suppression, scale_boxes
from ultralytics.data.augment import LetterBox

# ================= CONFIG =================
GT_DIR      = "/lab/projects/fire_smoke_awr/data/detection/test_sets/early_fire_test_clean"
YOLO_MODEL  = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/early_fire_pad_aug/train/weights/best.pt"
OUTPUT_DIR  = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/early_fire_pad_aug/test_set_clean/manual_resize_crop_inference"

IMG_SIZE = 640
INTERMEDIATE_SIZE = 800
SAVE_IMG = True
ANCHOR_FROM_GT = True  # Use GT box center instead of center crop
CONF_THRES = 0.25
IOU_THRES = 0.45
# ==========================================

IMAGE_DIR = os.path.join(GT_DIR, "images/test")
LABEL_DIR = os.path.join(GT_DIR, "labels/test")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_LABEL_DIR = os.path.join(OUTPUT_DIR, "labels")
SAVE_LB_DIR = os.path.join(OUTPUT_DIR, "letterbox")
SAVE_RESULT_DIR = os.path.join(OUTPUT_DIR, "results/annotated")

for d in [OUTPUT_DIR, SAVE_LABEL_DIR, SAVE_LB_DIR, SAVE_RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

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

    # Clamp to image bounds
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

    # Handle extreme cases
    crop_y1, crop_y2 = max(0, crop_y1), min(res_inter_h, crop_y2)
    crop_x1, crop_x2 = max(0, crop_x1), min(res_inter_w, crop_x2)

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


# ==========================================
if __name__ == "__main__":
    # --- Load PyTorch model directly ---
    ckpt = torch.load(YOLO_MODEL, map_location=DEVICE)
    model = ckpt["model"].to(DEVICE).float()
    model.eval()

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for img_name in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(IMAGE_DIR, img_name)
        base, _ = os.path.splitext(img_name)

        out_txt = os.path.join(SAVE_LABEL_DIR, base + ".txt")
        out_lb = os.path.join(SAVE_LB_DIR, base + "_letterbox.jpg")
        out_vis = os.path.join(SAVE_RESULT_DIR, base + "_annotated.jpg")

        original = cv2.imread(img_path)
        if original is None:
            continue

        # --- Determine object center ---
        label_path = os.path.join(LABEL_DIR, base + ".txt")
        if ANCHOR_FROM_GT:
            gt_center = load_gt_center(label_path)
            obj_center = gt_center if gt_center is not None else (0.5, 0.5)
        else:
            obj_center = (0.5, 0.5)

        # --- Generate crop ---
        cropped, meta = generate_crop_640x640(original, obj_center, INTERMEDIATE_SIZE)

        # --- Letterbox ---
        letterbox = LetterBox(IMG_SIZE, auto=False)
        img_lb = letterbox(image=cropped)

        # --- Prepare tensor ---
        im = img_lb[:, :, ::-1].transpose((2, 0, 1))  # BGR→RGB, HWC→CHW
        im = torch.from_numpy(np.ascontiguousarray(im)).float() / 255
        im = im.unsqueeze(0).to(DEVICE)

        # --- Inference ---
        with torch.no_grad():
            preds = model(im)  # raw model output

        # --- NMS ---
        preds = non_max_suppression(preds, conf_thres=CONF_THRES, iou_thres=IOU_THRES)[0]
        if preds is None or len(preds) == 0:
            continue

        boxes = preds[:, :4].cpu().numpy()
        confs = preds[:, 4].cpu().numpy()
        classes = preds[:, 5].cpu().numpy().astype(int)

        boxes_orig = scale_boxes(im.shape[2:], boxes, cropped.shape[:2]).round().astype(int)

        dets, mapped_preds = [], []
        for (x1, y1, x2, y2), conf, cls_id in zip(boxes_orig, confs, classes):
            mapped = yolo_to_original_crop_xyxy((x1, y1, x2, y2), meta, conf, cls_id)
            dets.append(mapped)
            mapped_preds.append(mapped)

        # --- Visualization ---
        if SAVE_IMG:
            cv2.imwrite(out_lb, img_lb)
            vis_image = cv2.resize(original, (meta["res_inter_w"], meta["res_inter_h"]))

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

            cv2.imwrite(out_vis, vis_image)

        # --- Save YOLO-format labels ---
        with open(out_txt, "w") as f:
            for d in dets:
                f.write(f"{d[0]} {d[1]:.6f} {d[2]:.6f} {d[3]:.6f} {d[4]:.6f} {d[5]:.4f}\n")

    print(f"✅ Finished processing {len(image_files)} images.")

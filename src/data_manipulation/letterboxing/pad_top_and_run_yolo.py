import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# ================ CONFIG =================
IMAGE_DIR   = "/lab/projects/fire_smoke_awr/data/test_sets/A/original/single_objects/images/test"
OUTPUT_DIR  = "/lab/projects/fire_smoke_awr/outputs/yolo/baseline/A_original_single_objects/pad_top_result"
YOLO_MODEL  = "/lab/projects/fire_smoke_awr/outputs/yolo/baseline/train/weights/best.pt"
TARGET_SIZE = 640
PADDING_VAL = (114, 114, 114)   # pad color
SAVE_PADDED = True              # save padded images for inspection
# ==========================================


def pad_bottom(image, new_shape=640, color=(114,114,114)):
    """Resize with aspect ratio, then bottom-anchor inside a 640x640 canvas."""
    h, w = image.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # create canvas
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)

    # put image at bottom center
    y_offset = new_shape - new_h
    x_offset = (new_shape - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    meta = {
        "scale": scale,
        "x_offset": x_offset,
        "y_offset": y_offset,
        "resized_w": new_w, "resized_h": new_h,
        "orig_w": w, "orig_h": h
    }
    return canvas, meta


def map_back(box_norm, conf, cls_id, meta):
    """Map YOLO normalized box back to original resolution."""
    xc, yc, w, h = box_norm
    x1 = (xc - w/2) * TARGET_SIZE
    y1 = (yc - h/2) * TARGET_SIZE
    x2 = (xc + w/2) * TARGET_SIZE
    y2 = (yc + h/2) * TARGET_SIZE

    # remove bottom-padding offset
    x1 -= meta["x_offset"]; x2 -= meta["x_offset"]
    y1 -= meta["y_offset"]; y2 -= meta["y_offset"]

    # scale back
    x1 /= meta["scale"]; x2 /= meta["scale"]
    y1 /= meta["scale"]; y2 /= meta["scale"]

    orig_w, orig_h = meta["orig_w"], meta["orig_h"]
    return [int(cls_id),
            ((x1 + x2) / 2) / orig_w,
            ((y1 + y2) / 2) / orig_h,
            (x2 - x1) / orig_w,
            (y2 - y1) / orig_h,
            float(conf)]


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if SAVE_PADDED:
        os.makedirs(os.path.join(OUTPUT_DIR, "padded_images"), exist_ok=True)

    model = YOLO(YOLO_MODEL)
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for img_name in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(IMAGE_DIR, img_name)
        base = os.path.splitext(img_name)[0]

        original = cv2.imread(img_path)
        if original is None:
            continue

        padded, meta = pad_bottom(original, TARGET_SIZE, PADDING_VAL)

        # save padded for inspection
        if SAVE_PADDED:
            cv2.imwrite(os.path.join(OUTPUT_DIR, "padded_images", img_name), padded)

        results = model.predict(padded, imgsz=TARGET_SIZE, conf=0.001, verbose=False)[0]

        dets = []
        for box, conf, cls_id in zip(results.boxes.xywhn.cpu().numpy(),
                                     results.boxes.conf.cpu().numpy(),
                                     results.boxes.cls.cpu().numpy()):
            mapped = map_back(box, conf, cls_id, meta)
            dets.append(mapped)

        # save results
        out_path = os.path.join(OUTPUT_DIR, base + ".txt")
        with open(out_path, "w") as f:
            for cls_id, xc, yc, w, h, conf in dets:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.4f}\n")

    print(f"✅ Finished processing {len(image_files)} images. Results saved to {OUTPUT_DIR}")

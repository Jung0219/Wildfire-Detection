import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from glob import glob
from ultralytics.utils.ops import non_max_suppression, scale_boxes
from ultralytics.data.augment import LetterBox

# --- CONFIG ---
GT_DIR = "/lab/projects/fire_smoke_awr/data/detection/test_sets/early_fire_test_clean"
YOLO_MODEL = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/early_fire_pad_aug/train/weights/best.pt"
OUTPUT_DIR = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/early_fire_pad_aug/test_set_clean/manual_resize_original_inference"
IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.45
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- SETUP ---
IMAGE_DIR = os.path.join(GT_DIR, "images/test")
LABEL_DIR = os.path.join(GT_DIR, "labels/test")
SAVE_LABEL_DIR = os.path.join(OUTPUT_DIR, "labels")
SAVE_LB_DIR = os.path.join(OUTPUT_DIR, "letterbox")
SAVE_RESULT_DIR = os.path.join(OUTPUT_DIR, "results/annotated")

for d in [OUTPUT_DIR, SAVE_LABEL_DIR, SAVE_LB_DIR, SAVE_RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

# --- LOAD PYTORCH MODEL DIRECTLY ---
ckpt = torch.load(YOLO_MODEL, map_location=DEVICE)
model = ckpt["model"].to(DEVICE).float()
model.eval()

# --- LOOP OVER IMAGES ---
for img_path in tqdm(sorted(glob(os.path.join(IMAGE_DIR, "*.jpg"))), desc="Processing images"):
    basename = os.path.splitext(os.path.basename(img_path))[0]
    save_img_path = os.path.join(SAVE_RESULT_DIR, f"{basename}_result.jpg")
    save_lb_path = os.path.join(SAVE_LB_DIR, f"{basename}_letterbox.jpg")
    save_label_path = os.path.join(SAVE_LABEL_DIR, f"{basename}.txt")

    # --- Load image ---
    img0 = cv2.imread(img_path)
    if img0 is None:
        print(f"[!] Skipped invalid image: {img_path}")
        continue
    orig_shape = img0.shape[:2]

    # --- Letterbox resize ---
    letterbox = LetterBox(IMG_SIZE, auto=False)
    img_lb = letterbox(image=img0)

    # --- To tensor ---
    im = img_lb[:, :, ::-1].transpose((2, 0, 1))  # BGR->RGB, HWC->CHW
    im = torch.from_numpy(np.ascontiguousarray(im)).float() / 255
    im = im.unsqueeze(0).to(DEVICE)

    # --- Inference ---
    with torch.no_grad():
        preds = model(im)  # raw predictions

    # --- NMS ---
    preds = non_max_suppression(preds, conf_thres=CONF_THRES, iou_thres=IOU_THRES)[0]
    if preds is None or len(preds) == 0:
        continue

    # --- Extract outputs ---
    boxes = preds[:, :4].cpu().numpy()
    confs = preds[:, 4].cpu().numpy()
    classes = preds[:, 5].cpu().numpy().astype(int)

    # --- Scale boxes back to original image ---
    boxes_orig = scale_boxes(im.shape[2:], boxes, orig_shape).round().astype(int)

    # --- Draw boxes ---
    img_drawn = img0.copy()
    for (x1, y1, x2, y2), conf, cls in zip(boxes_orig, confs, classes):
        cv2.rectangle(img_drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{cls} {conf:.2f}"
        cv2.putText(img_drawn, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --- Save results ---
    cv2.imwrite(save_lb_path, img_lb)
    cv2.imwrite(save_img_path, img_drawn)

    # --- Save YOLO-format labels ---
    with open(save_label_path, "w") as f:
        for (x1, y1, x2, y2), conf, cls in zip(boxes_orig, confs, classes):
            x_center = (x1 + x2) / 2 / orig_shape[1]
            y_center = (y1 + y2) / 2 / orig_shape[0]
            w = (x2 - x1) / orig_shape[1]
            h = (y2 - y1) / orig_shape[0]
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {conf:.4f}\n")

print("\nAll results saved to:")
print(" -", SAVE_RESULT_DIR)
print(" -", SAVE_LB_DIR)
print(" -", SAVE_LABEL_DIR)

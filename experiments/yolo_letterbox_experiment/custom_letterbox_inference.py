import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.ops import scale_boxes
from ultralytics.data.augment import LetterBox  # <--- import LetterBox

# --- CONFIG ---
IMAGE_PATH = "/lab/projects/fire_smoke_awr/data/detection/test_sets/early_fire/test/images/test/smoke_UAV000851.jpg"
YOLO_MODEL_PATH = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/early_fire_pad_aug/train/weights/best.pt"
SAVE_DIR = "/lab/projects/fire_smoke_awr/experiments/yolo_letterbox_experiment/results"
IMG_SIZE = 640

# --- 1. Setup ---
os.makedirs(SAVE_DIR, exist_ok=True)
basename = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

# --- 2. Load Image ---
img0 = cv2.imread(IMAGE_PATH)
orig_shape = img0.shape[:2]

# --- 3. Apply Ultralytics Letterbox Transform ---
transform = LetterBox(IMG_SIZE, auto=False)
img_lb = transform(image=img0)  # returns padded image only
# (The transform stores scale and padding info internally)

# --- 4. Convert to Tensor ---
im = img_lb[:, :, ::-1].transpose((2, 0, 1))  # BGR->RGB, HWC->CHW
im = torch.from_numpy(np.ascontiguousarray(im)).float() / 255
im = im.unsqueeze(0)

# --- 5. Inference ---
model = YOLO(YOLO_MODEL_PATH)
results = model(im, verbose=True)

# --- 6. Rescale Boxes ---
boxes = results[0].boxes.xyxy.cpu().numpy()
boxes_orig = scale_boxes(im.shape[2:], boxes, orig_shape).round().astype(int)

# --- 7. Draw and Save ---
img_drawn = img0.copy()
boxes = results[0].boxes.xyxy.cpu().numpy()
confidences = results[0].boxes.conf.cpu().numpy()  # <--- confidence values

for (x1, y1, x2, y2), conf in zip(boxes_orig, confidences):
    cv2.rectangle(img_drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{conf:.2f}"  # show confidence up to 2 decimals
    cv2.putText(img_drawn, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imwrite(os.path.join(SAVE_DIR, f"{basename}_letterbox.jpg"), img_lb)
cv2.imwrite(os.path.join(SAVE_DIR, f"{basename}_result.jpg"), img_drawn)
print("[✓] Results saved to", SAVE_DIR)
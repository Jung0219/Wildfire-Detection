import os
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.ops import letterbox

# --------------------- CONFIGURATION ---------------------
MODEL_PATH = "/lab/projects/fire_smoke_awr/outputs/yolo/deduplicated/phash6/A+B+C+D/train/weights/best.pt"
INPUT_DIR = "/lab/projects/fire_smoke_awr/src/util/experiment/letterboxed"
OUTPUT_DIR = "/lab/projects/fire_smoke_awr/src/util/experiment/letterboxed/results"
CONF_THRESHOLD = 0.25
IMG_SIZE = 640
STRIDE = 32
# ---------------------------------------------------------

# Load model
model = YOLO(MODEL_PATH)
device = model.device

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Valid image extensions
valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Inference loop
image_files = list(Path(INPUT_DIR).glob("*"))
for img_path in image_files:
    if img_path.suffix.lower() not in valid_exts:
        continue

    # Load original BGR image
    img0 = cv2.imread(str(img_path))
    if img0 is None:
        print(f"Failed to load: {img_path}")
        continue

    # Manual YOLO-style letterbox and preprocessing
    img, _, _ = letterbox(img0, new_shape=IMG_SIZE, stride=STRIDE, auto=True)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB -> CHW
    img = np.ascontiguousarray(img).astype(np.float32) / 255.0  # Normalize
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)  # (1, 3, H, W)

    # Run inference
    preds = model.predict(source=img_tensor, conf=CONF_THRESHOLD, stream=True, imgsz=IMG_SIZE)
    results = next(preds)

    # Draw detections on original image
    for box in results.boxes:
        xyxy = box.xyxy.cpu().numpy().astype(int)[0]
        conf = float(box.conf)
        cls_id = int(box.cls)
        label = model.names[cls_id]
        text = f"{label} {conf:.2f}"

        cv2.rectangle(img0, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        cv2.putText(img0, text, (xyxy[0], xyxy[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Save annotated image
    out_path = Path(OUTPUT_DIR) / img_path.name
    cv2.imwrite(str(out_path), img0)

print(f"Inference complete. Annotated images saved to: {OUTPUT_DIR}")

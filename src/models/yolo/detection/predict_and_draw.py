import os
import cv2
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# ---------------- CONFIG ----------------
MODEL_PATH  = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/early_fire_pad_aug/train/weights/best.pt"
IMAGE_DIR   = "/lab/projects/fire_smoke_awr/src/data_manipulation/translation_equivariance_experiment/experiment5/images"
SAVE_DIR    = "/lab/projects/fire_smoke_awr/src/data_manipulation/translation_equivariance_experiment/experiment5/results"
CONF_THRESHOLD = 0.25  # confidence threshold
IOU_THRESHOLD  = 0.45  # NMS threshold
# ----------------------------------------


def run_yolo_inference(model_path, image_dir, save_dir, conf_thres=0.25, iou_thres=0.45):
    model = YOLO(model_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    image_files = [p for p in Path(image_dir).rglob("*") if p.suffix.lower() in valid_exts]

    print(f"Found {len(image_files)} images. Running inference...")

    for img_path in tqdm(image_files, desc="Processing", unit="img"):
        results = model.predict(
            source=str(img_path),
            conf=conf_thres,
            iou=iou_thres,
            verbose=False
        )[0]

        img = cv2.imread(str(img_path))
        if img is None:
            tqdm.write(f"[WARN] Skipped unreadable image: {img_path}")
            continue

        h, w = img.shape[:2]

        for box in results.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = f"{model.names[cls_id]} {conf:.2f}"
            color = (0, 255, 0)

            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

            # Compute label size
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Preferred position: above the box
            text_x = max(0, min(x1, w - text_w))
            text_y = y1 - 5

            # If text would go above the image, move it below
            if text_y - text_h < 0:
                text_y = min(y2 + text_h + 5, h - baseline)

            # Draw filled background for visibility
            cv2.rectangle(
                img,
                (text_x, text_y - text_h - baseline),
                (text_x + text_w, text_y + baseline // 2),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                img, label, (text_x, text_y - baseline),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
            )

        out_path = save_dir / img_path.name
        cv2.imwrite(str(out_path), img)

    print(f"✅ Done! Annotated results saved to: {save_dir}")


if __name__ == "__main__":
    run_yolo_inference(
        MODEL_PATH,
        IMAGE_DIR,
        SAVE_DIR,
        conf_thres=CONF_THRESHOLD,
        iou_thres=IOU_THRESHOLD
    )

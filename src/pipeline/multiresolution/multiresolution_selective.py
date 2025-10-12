import os
import argparse
import yaml
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from composite_utils import process_image

# ==================== CONFIG (default values) ====================
# Directories (will be overridden by YAML if provided)
IMAGE_DIR   = "/lab/projects/fire_smoke_awr/data/detection/training/deduplicated/phash10/A+B+C+D+E/split/images/test"  
OUTPUT_DIR  = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/ABCDE_phash10/test_set" 
MODEL_PATH = os.path.join(os.path.dirname(OUTPUT_DIR), "train", "weights", "best.pt")

# Processing options
SAVE_IMG        = False
INTERMEDIATE_SZ = 780
NMS_IOU_THRESH  = 0.45
POSTPROC        = "nms"
CONF_THRESH     = 0.001

# Composite condition thresholds
Y_MIN = 80          # Minimum Y value (luminance)
Y_MAX = 255  
CB_MIN = 0         # Minimum Cb value for boundary
CB_MAX = 140        # Maximum Cb value for boundary
CR_MIN = 150         # Minimum Cr value for boundary
CR_MAX = 255         # Maximum Cr value for boundary
RATIO_TH = 0.05   # use composite if fire pixel ratio < 5%
# ================================================================

def use_composite(img_bgr,
                  cb_min=CB_MIN, cb_max=CB_MAX,
                  cr_min=CR_MIN, cr_max=CR_MAX,
                  y_min=Y_MIN, y_max=Y_MAX,
                  ratio_thresh=RATIO_TH):
    """
    Use composite only if fire pixel ratio < ratio_thresh
    => possible early fire scene
    """
    H, W = img_bgr.shape[:2]
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    m_full = ((Y >= y_min) & (Y <= y_max) &
              (Cb >= cb_min) & (Cb <= cb_max) &
              (Cr >= cr_min) & (Cr <= cr_max)).astype(np.uint8)

    ratio = m_full.mean()
    return ratio < ratio_thresh


def main():
    # Subdirectories
    composite_dir = os.path.join(OUTPUT_DIR, "composites_selective_images")
    txt_dir = os.path.join(OUTPUT_DIR, "composites_selective")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    if SAVE_IMG:
        os.makedirs(composite_dir, exist_ok=True)

    # Load YOLO model
    model = YOLO(MODEL_PATH)

    # Gather images
    image_files = [f for f in os.listdir(IMAGE_DIR)
                   if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    # Process images
    for img_name in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(IMAGE_DIR, img_name)
        base, _ = os.path.splitext(img_name)

        out_txt = os.path.join(txt_dir, base + ".txt")
        out_composite = os.path.join(composite_dir, base + "_composite.jpg")

        # Read the image once
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        if use_composite(img_bgr):
            # Run composite pipeline
            process_image(
                img_path=img_path,
                model=model,
                out_txt=out_txt,
                out_composite=out_composite,
                intermediate_size=INTERMEDIATE_SZ,
                nms_iou_thresh=NMS_IOU_THRESH,
                postproc=POSTPROC
            )
        else:
            # Run plain YOLO inference
            results = model.predict(
                source=img_path,
                conf=CONF_THRESH,
                verbose=False
            )
            # Save detections in YOLO txt format
            r = results[0]
            with open(out_txt, "w") as f:
                for box in r.boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    x_center, y_center, w, h = box.xywhn[0].tolist()  # already normalized
                    f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")

    print(f"\n✅ Finished processing {len(image_files)} images.")
    print(f"Results saved under {txt_dir}")


if __name__ == "__main__":
    main()

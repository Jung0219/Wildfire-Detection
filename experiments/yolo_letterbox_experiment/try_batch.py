import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from letterbox import LetterBox

# ================= CONFIG =================
IMAGE_DIR = "/lab/projects/fire_smoke_awr/data/detection/test_sets/ABCDE_noEF_10%/images/test"
YOLO_MODEL_PATH = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/ABCDE_noEF/train/weights/best.pt"

OUTPUT_DIR = "/lab/projects/fire_smoke_awr/src/data_manipulation/letterboxing/yolo_letterbox_experiment/results"
OUTPUT_DIR_ORIG = os.path.join(OUTPUT_DIR, "outputs/original_pred")
OUTPUT_DIR_LB = os.path.join(OUTPUT_DIR, "outputs/letterboxed_pred")

os.makedirs(OUTPUT_DIR_ORIG, exist_ok=True)
os.makedirs(OUTPUT_DIR_LB, exist_ok=True)
# ==========================================

# Initialize YOLO and LetterBox
model = YOLO(YOLO_MODEL_PATH)
letterbox = LetterBox(new_shape=(640, 640), auto=False, scale_fill=False, scaleup=False)

# ============ HELPERS ============

def draw_predictions(image, boxes, color=(0, 255, 0)):
    """Draw bounding boxes on image."""
    if hasattr(boxes, "xyxy"):
        boxes_iter = boxes
    else:
        class DummyBox:
            def __init__(self, coords):
                self.xyxy = np.array([coords])
                self.conf = [1.0]
                self.cls = [0]
        boxes_iter = [DummyBox(b) for b in boxes]

    for box in boxes_iter:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = getattr(box, "conf", [1.0])[0]
        cls = int(getattr(box, "cls", [0])[0])
        label = f"{model.names[cls]} {conf:.2f}" if hasattr(model, "names") else f"{cls} {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        cv2.putText(image, label, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image

def iou_xyxy(box_a, box_b, eps=1e-9):
    """Compute IoU between two xyxy boxes."""
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return float(inter_area / (union + eps))


# ============ MAIN LOOP ============

ious = []
conf_diffs = []

for img_name in tqdm(sorted(os.listdir(IMAGE_DIR)), desc="Processing Images"):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping invalid image: {img_path}")
        continue

    # 1️⃣ Regular YOLO inference
    results_orig = model(img, verbose=False)[0]
    img_drawn_orig = draw_predictions(img.copy(), results_orig.boxes, color=(0, 255, 0))
    cv2.imwrite(os.path.join(OUTPUT_DIR_ORIG, img_name), img_drawn_orig)

    # 2️⃣ Letterboxed inference
    lb_img = letterbox(image=img)
    results_lb = model(lb_img, verbose=False)[0]
    img_drawn_lb = draw_predictions(lb_img.copy(), results_lb.boxes, color=(255, 255, 0))
    cv2.imwrite(os.path.join(OUTPUT_DIR_LB, img_name), img_drawn_lb)

    # Extract boxes & confs
    orig_boxes = results_orig.boxes.xyxy.cpu().numpy()
    lb_boxes = results_lb.boxes.xyxy.cpu().numpy()
    lb_boxes_mapped = LetterBox.undo_letterbox(
        boxes=lb_boxes.copy(),
        orig_shape=img.shape[:2],
        new_shape=lb_img.shape[:2],
        scaleup=False,
        center=True
    )

    orig_conf = results_orig.boxes.conf.cpu().numpy()
    lb_conf = results_lb.boxes.conf.cpu().numpy()

    min_len = min(len(orig_boxes), len(lb_boxes_mapped))
    if min_len == 0:
        continue

    # Compute IoUs & conf differences
    img_ious = []
    img_conf_diffs = []
    for i in range(min_len):
        iou_val = iou_xyxy(orig_boxes[i], lb_boxes_mapped[i])
        img_ious.append(iou_val)
        img_conf_diffs.append(lb_conf[i] - orig_conf[i])

    ious.extend(img_ious)
    conf_diffs.extend(img_conf_diffs)

# ============ SUMMARY ============
avg_iou = np.mean(ious) if ious else 0
avg_conf_diff = np.mean(conf_diffs) if conf_diffs else 0

print("\n==================== RESULTS SUMMARY ====================")
print(f"Average IoU: {avg_iou:.4f}")
print(f"Average Confidence Difference (Letterboxed - Original): {avg_conf_diff:+.4f}")
print(f"==========================================================")
print(f"Results saved to:\n- {OUTPUT_DIR_ORIG}\n- {OUTPUT_DIR_LB}")

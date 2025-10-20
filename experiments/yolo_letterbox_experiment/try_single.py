import os
import cv2
import numpy as np
from ultralytics import YOLO
from letterbox import LetterBox

# ================= CONFIG =================
IMAGE_PATH = "/lab/projects/fire_smoke_awr/data/detection/test_sets/early_fire/test/images/test/smoke_UAV000851.jpg"  # <-- change this
YOLO_MODEL_PATH = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/early_fire_pad_aug/train/weights/best.pt"

OUTPUT_DIR = "/lab/projects/fire_smoke_awr/src/data_manipulation/letterboxing/yolo_letterbox_experiment/results/debug"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ==========================================

# Initialize YOLO and LetterBox
model = YOLO(YOLO_MODEL_PATH)
letterbox = LetterBox(new_shape=(640, 640), auto=False, scale_fill=False, scaleup=False)

def draw_predictions(image, boxes, color=(0, 255, 0)):
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
    """
    Compute IoU between two boxes in xyxy format.
    box_a, box_b: iterable of [x1, y1, x2, y2]
    Returns float IoU in [0, 1].
    """
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)

    # intersection
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # areas
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return float(inter_area / (union + eps))

# ===== Single Image Debug Run =====
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError(f"Invalid image: {IMAGE_PATH}")

# 1️⃣ Regular YOLO inference
results_orig = model(img, verbose=True)[0]
img_drawn_orig = draw_predictions(img.copy(), results_orig.boxes, color=(0, 255, 0))
cv2.imwrite(os.path.join(OUTPUT_DIR, "original_pred.jpg"), img_drawn_orig)

# 2️⃣ YOLO inference on letterboxed image
lb_img = letterbox(image=img)
results_lb = model(lb_img, verbose=True)[0]

img_drawn_orig = draw_predictions(lb_img.copy(), results_lb.boxes, color=(0, 255, 0))
cv2.imwrite(os.path.join(OUTPUT_DIR, "letterbox_pred.jpg"), img_drawn_orig)


boxes_xyxy = results_lb.boxes.xyxy.cpu().numpy()

boxes_mapped = LetterBox.undo_letterbox(
    boxes=boxes_xyxy.copy(),
    orig_shape=img.shape[:2],     # (h, w) of the original image
    new_shape=lb_img.shape[:2],   # (h, w) of the letterboxed image
    scaleup=False,
    center=True 
)

orig_conf = results_orig.boxes.conf.cpu().numpy()
lb_conf = results_lb.boxes.conf.cpu().numpy()

print(f"  Original Conf: {orig_conf}, Letterboxed Conf   : {lb_conf}")
print(f"  Difference: {lb_conf - orig_conf}")

orig_box = results_orig.boxes.xyxy.cpu().numpy()
lb_box = boxes_mapped

print(orig_box, lb_box)

print("iou between boxes:", iou_xyxy(orig_box[0], lb_box[0]))
print(f"\n✅ Debug complete!\nSaved results in: {OUTPUT_DIR}")


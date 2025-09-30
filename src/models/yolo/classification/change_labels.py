import os
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm

# ==============================
# CONFIGURATION
# ==============================
IMAGES_DIR = "/lab/projects/fire_smoke_awr/data/detection/test_sets/early_fire/images/test"        
PREDICTIONS_DIR = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/new_model/early_fire/composites"    
CLASSIFIER_MODEL = "/lab/projects/fire_smoke_awr/outputs/yolo/classification/ABCDE_early_fire_removed/train/weights/best.pt"  
OUTPUT_DIR = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/new_model/early_fire/cls_fixed"  
IMG_SIZE = 224                        
# ==============================


def load_classifier(model_path):
    return YOLO(model_path)


def crop_from_yolo(image, bbox, img_w, img_h):
    """Convert YOLO bbox (x_center, y_center, w, h) to pixel crop."""
    x_c, y_c, w, h = bbox
    x_c, y_c, w, h = x_c * img_w, y_c * img_h, w * img_w, h * img_h
    x1 = max(int(x_c - w / 2), 0)
    y1 = max(int(y_c - h / 2), 0)
    x2 = min(int(x_c + w / 2), img_w - 1)
    y2 = min(int(y_c + h / 2), img_h - 1)
    return (x1, y1, x2, y2)


def classify_crop(model, crop_img):
    results = model(crop_img, imgsz=IMG_SIZE, verbose=False)[0]
    cls_id = int(results.probs.top1)
    # Map ternary classifier → binary labels
    if cls_id == 0:   # background
        return 0
    else:             # fire or smoke
        return 1


def process_predictions(images_dir, preds_dir, classifier, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model = load_classifier(classifier)

    txt_files = [f for f in os.listdir(preds_dir) if f.endswith(".txt")]

    for txt_file in tqdm(txt_files, desc="Processing predictions"):
        txt_path = os.path.join(preds_dir, txt_file)
        img_name = os.path.splitext(txt_file)[0] + ".jpg"  # adjust if png
        img_path = os.path.join(images_dir, img_name)

        if not os.path.exists(img_path):
            print(f"Image not found for {txt_file}, skipping...")
            continue

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        new_lines = []
        with open(txt_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                # YOLO format: class x_center y_center width height [conf]
                cls, x_c, y_c, w, h, *rest = parts
                bbox = (float(x_c), float(y_c), float(w), float(h))

                # crop
                x1, y1, x2, y2 = crop_from_yolo(img, bbox, img_w, img_h)
                crop_img = img.crop((x1, y1, x2, y2))

                # classify (mapped to binary)
                new_cls = classify_crop(model, crop_img)

                # rebuild line
                new_line = " ".join([str(new_cls), x_c, y_c, w, h] + rest)
                new_lines.append(new_line)

        # save to new directory
        out_path = os.path.join(output_dir, txt_file)
        with open(out_path, "w") as f:
            f.write("\n".join(new_lines))


if __name__ == "__main__":
    process_predictions(IMAGES_DIR, PREDICTIONS_DIR, CLASSIFIER_MODEL, OUTPUT_DIR)

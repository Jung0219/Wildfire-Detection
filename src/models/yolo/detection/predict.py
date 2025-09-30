import os
from ultralytics import YOLO

# ================= CONFIGURATION =================
MODEL_PATH = "/lab/projects/fire_smoke_awr/outputs/yolo/final/train/weights/best.pt"
IMAGE_DIR = "/lab/projects/fire_smoke_awr/data/final/test_data/small_object_set/hand-filtered/composites/images"
RESULT_DIR = "/lab/projects/fire_smoke_awr/data/final/test_data/small_object_set/hand-filtered/composites/predictions"
# =================================================

# Load the trained model
model = YOLO(MODEL_PATH)

# Run prediction
model.predict(
    source=IMAGE_DIR,
    save=False,
    save_txt=True,
    save_conf=True,
    project=os.path.dirname(RESULT_DIR),
    name=os.path.basename(RESULT_DIR),
    exist_ok=True,
    conf=0.001
)

# Ensure all images have corresponding .txt files
print("🛠 Ensuring all images have annotation files (creating empty ones if needed)...")
labels_dir = os.path.join(RESULT_DIR, "labels")
os.makedirs(labels_dir, exist_ok=True)

image_basenames = [
    os.path.splitext(f)[0]
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

for base in image_basenames:
    txt_path = os.path.join(labels_dir, base + ".txt")
    if not os.path.exists(txt_path):
        open(txt_path, 'w').close()

print(f"✅ Inference complete. Results saved to: {labels_dir}")

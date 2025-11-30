from datetime import datetime
from ultralytics import YOLO

# ==== CONFIGURATION ====
MODEL_PATH = "yolov8s-cls.pt"
# ======================
DATA_DIR = "/lab/projects/fire_smoke_awr/data/classification/training/AD_phash3_early_smoke"    
# ^ dataset root containing train/, val/, test/
PROJECT_DIR = "/lab/projects/fire_smoke_awr/outputs/yolo/classification/AD_phash3_early_smoke"
EPOCHS = 100
IMG_SIZE = 224
BATCH = 32
# ========================

# Load YOLO classification model
model = YOLO(MODEL_PATH)

start_time = datetime.now()
print(f"[INFO] Training started at {start_time}")

# Train
model.train(
    data=DATA_DIR,        # root folder with train/ and val/
    project=PROJECT_DIR,  # training logs/checkpoints
    name="train",
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    resume=False,
)


end_time = datetime.now()
print(f"[INFO] Training started at {start_time}")
print(f"[INFO] Finished at {end_time}")
print(f"[INFO] Total time: {end_time - start_time}")

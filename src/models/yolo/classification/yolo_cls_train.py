from datetime import datetime
from ultralytics import YOLO

# ==== CONFIGURATION ====
MODEL_PATH = "yolov8s-cls.pt"
# ======================
DATA_DIR = "/lab/projects/fire_smoke_awr/data/classification/training/ABCDE_early_fire_removed"    
# ^ dataset root containing train/, val/, test/
PROJECT_DIR = "/lab/projects/fire_smoke_awr/outputs/yolo/classification/ABCDE_early_fire_removed"
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

    # --- Augmentations (classification-focused) ---
    #liplr=0.5,        # horizontal flip
    #degrees=10.0,      # small rotation
    #scale=0.1,         # mild scaling
    #translate=0.05,    # light translation
    #hsv_h=0.0,         # no hue jitter
    #hsv_s=0.0,         # no saturation jitter
    #hsv_v=0.2,         # mild brightness jitter

    # --- Disable others ---
    #mosaic=0.0,
    #mixup=0.0,
    #cutmix=0.0,
    #copy_paste=0.0,
    #auto_augment=None,
    #erasing=0.0
)


end_time = datetime.now()
print(f"[INFO] Training started at {start_time}")
print(f"[INFO] Finished at {end_time}")
print(f"[INFO] Total time: {end_time - start_time}")

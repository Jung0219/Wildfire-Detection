from ultralytics import YOLO
import os
from pathlib import Path

# Define paths
parent_dir = '/lab/projects/fire_smoke_awr/outputs/yolo/detection/early_fire_letterbox' 
run_name = 'early_fire_normal_test_set'

# Load the trained model
model_path = os.path.join(parent_dir, 'train/weights/best.pt')
print(f"[INFO] Loading model from: {model_path}")
model = YOLO(model_path)

# Run evaluation and save predictions
results = model.val(
    data='/lab/projects/fire_smoke_awr/src/models/yolo/detection/test.yaml',
    project=parent_dir,
    name=run_name,
    save=True,
    save_txt=True,
    save_conf=True,
    verbose=True,
    cache=False,
    conf=0.001
)
from datetime import datetime
from ultralytics import YOLO

project_dir = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/early_smoke_A"  # change as needed

model = YOLO("/lab/projects/fire_smoke_awr/weights/pretrained/yolov8/yolov8s.pt")  # change to yolov8s.pt, m.pt, etc. as needed
start_time = datetime.now()
print(f"[INFO] Training started at {start_time}")
# Train 
model.train( 
    data='/lab/projects/fire_smoke_awr/src/models/yolo/detection/train.yaml',
    project=project_dir,
    name="train",
    epochs=100,
    imgsz=640,
    batch=16,                       # Subfolder for training results
    resume=False
)
 
end_time = datetime.now()
print(f"[INFO] Training started at {start_time}")
print(f"[INFO] Finished at {end_time}")
print(f"[INFO] Total time: {end_time - start_time}")
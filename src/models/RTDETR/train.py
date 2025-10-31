from datetime import datetime
from ultralytics import RTDETR

project_dir = "/lab/projects/fire_smoke_awr/outputs/RTDETR/early_fire_crop_aug"  # change as needed

model = RTDETR("/lab/projects/fire_smoke_awr/weights/detection/RTDETR/rtdetr-l.pt")  # change to yolov8s.pt, m.pt, etc. as needed
start_time = datetime.now()
print(f"[INFO] Training started at {start_time}")
# Train 
model.train( 
    data='/lab/projects/fire_smoke_awr/src/models/RTDETR/train.yaml',
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
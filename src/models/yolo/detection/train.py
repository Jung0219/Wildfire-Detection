from datetime import datetime
from ultralytics import YOLO

project_dir = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/early_fire_res_diff/896"

# Fine-tune pretrained YOLOv8s on the >=1080px filtered dataset.
model = YOLO("/lab/projects/fire_smoke_awr/weights/detection/yolov8/yolov8s.pt")

start_time = datetime.now()
print(f"[INFO] Training started at {start_time}")

model.train(
<<<<<<< HEAD
    data="/lab/projects/fire_smoke_awr/src/models/yolo/detection/train.yaml",
    project=project_dir,
    name="train",
    epochs=100,
    imgsz=896,
    batch=16,
    resume=False,
=======
    data="/lab/biohpc/ComputerVisionAI/fire_smoke_awr/src/models/yolo/detection/train.yaml",
    project=project_dir,
    name="train",
    epochs=100,
    imgsz=640,
    batch=16,
    optimizer="AdamW",
    patience=15,
    resume=False,
    device=4
>>>>>>> 20e489c07ba06d3fcc44f6bcb36693049d328deb
)

end_time = datetime.now()
print(f"[INFO] Training started at {start_time}")
print(f"[INFO] Finished at {end_time}")
print(f"[INFO] Total time: {end_time - start_time}")

import os
import fiftyone as fo
from fiftyone import Sample

# Path to your image directory
image_dir = "/lab/projects/fire_smoke_awr/src/data_manipulation/letterboxing/yolo_letterbox_experiment/results/outputs/original_pred"

# Create a new dataset
dataset_name = "orignal"

if dataset_name in fo.list_datasets():
    print("deleting duplicate ...")
    fo.delete_dataset(dataset_name)

# Create new dataset
dataset = fo.Dataset(name=dataset_name, persistent=True)
# Collect image paths
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
               if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# Create and add samples
samples = [Sample(filepath=path) for path in image_paths]
dataset.add_samples(samples)

# Launch session
session = fo.launch_app(dataset)
session.wait()

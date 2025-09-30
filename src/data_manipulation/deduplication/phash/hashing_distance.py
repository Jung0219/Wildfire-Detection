import os
from PIL import Image
import imagehash
import pandas as pd
from tqdm import tqdm

# Parameters
image_dir = "/lab/projects/fire_smoke_awr/data/original/FireMan_UAV_RGBT/resized/224x224/images"
max_frames = 1000  # Number of frames to compare (frame 1 vs frame 2 to N)

# Sorted list of image paths
image_list = [
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
]

# Limit the list to desired comparison range
image_list = image_list[430:5440]

# Load reference image (frame 1)
ref_img_path = image_list[4512]
ref_img = Image.open(ref_img_path).convert("L")
ref_hash = imagehash.phash(ref_img)

# Compare to subsequent frames
records = []
for i in tqdm(range(1, len(image_list))):
    img_path = image_list[i]
    img = Image.open(img_path).convert("L")
    img_hash = imagehash.phash(img)
    distance = ref_hash - img_hash

    records.append({
        "Image Name": os.path.basename(img_path),
        "Hamming Distance to Frame 1": distance
    })

# Create and display table
df = pd.DataFrame(records)
df.index += 2  # Frame numbering (starts from frame 2)

df.to_csv("hamming_distances.csv", index=False)
print("Saved hamming_distances.csv")

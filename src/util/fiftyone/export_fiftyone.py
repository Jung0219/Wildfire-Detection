import fiftyone as fo
from fiftyone import ViewField as F

export_dir = "/lab/projects/fire_smoke_awr/data/detection/datasets/data_mining/single_objects_lt_130/dedup_phash5/early_fire_hand_selected/images/test"

# --- Step 1: Load dataset ---
dataset_name = "ABCDE_single_objects_lt_130_dedup_phash5"
dataset = fo.load_dataset(dataset_name)

# --- Step 2: Show available tags ---
available_tags = dataset.distinct("tags")
print("Available sample tags:", available_tags)

# --- Step 3: Ask user which tag(s) to export ---
chosen_tags = input("Enter the tag(s) you want to export (comma-separated): ").split(",")
chosen_tags = [t.strip() for t in chosen_tags if t.strip()]

# --- Step 4: Get samples that have those tags ---
tagged_view = dataset.match_tags(chosen_tags)

# --- Step 5: Count them ---
print(f"Number of samples with tags {chosen_tags}: {len(tagged_view)}")

# --- Step 6: Export tagged images ---
tagged_view.export(
    export_dir=export_dir,
    dataset_type=fo.types.ImageDirectory,
)

print(f"Exported {len(tagged_view)} tagged images to {export_dir}")

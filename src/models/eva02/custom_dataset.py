import os
import json
from PIL import Image
from torch.utils.data import Dataset

class CustomJsonDataset(Dataset):
    def __init__(self, img_dir, json_path, transform=None):
        """
        Args:
            img_dir (str): Path to folder containing all images.
            json_path (str): Path to JSON file mapping {filename: label}.
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        self.img_dir = img_dir
        self.transform = transform

        # Load mapping from JSON
        with open(json_path, "r") as f:
            self.mapping = json.load(f)

        # Build list of (filename, label) pairs
        self.samples = list(self.mapping.items())

        # Unique classes and mapping to indices
        self.classes = sorted(set(self.mapping.values()))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label_name = self.samples[idx]
        img_path = os.path.join(self.img_dir, fname)

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Map label string to integer
        label = self.class_to_idx[label_name]

        return image, label

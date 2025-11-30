"""
Classifier wrappers used in two-stage pipelines.

Example:
    python -m src.pipeline.two_stage.classify_region
"""

import numpy as np

class BaseClassifier:
    def predict(self, crop_image: np.ndarray) -> str:
        """
        Input: crop_image as numpy array (H, W, 3)
        Output: string label, e.g. "background" or "fire"
        """
        raise NotImplementedError


# ===== YOLO Wrapper =====
from ultralytics import YOLO

class YOLOClassifier(BaseClassifier):
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)

    def predict(self, crop_image):
        results = self.model.predict(crop_image, verbose=False)[0]

        if results.probs is not None:
            cls_id = int(results.probs.top1)
            return results.names[cls_id]

        boxes = results.boxes
        if boxes is None or boxes.data.numel() == 0:
            return "background"

        if boxes.conf is not None:
            best_idx = int(boxes.conf.argmax())
        else:
            best_idx = 0
        cls_id = int(boxes.cls[best_idx])
        names = results.names
        if isinstance(names, dict):
            return names.get(cls_id, str(cls_id))
        if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
            return names[cls_id]
        return str(cls_id)


# ===== EVA Wrapper =====
import torch
from torchvision import transforms
from src.models.eva02.eva02_model import EVA02Classifier  # adjust path if needed

class EVAClassifier(BaseClassifier):
    def __init__(self, weights_path, device="cuda"):
        self.model = EVA02Classifier(num_classes=2)  # adapt args to your EVA model
        state = torch.load(weights_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.to(device).eval()
        self.device = device

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])

    def predict(self, crop_image):
        if crop_image is None or crop_image.size == 0:
            return "background"
        tensor = self.transform(crop_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            pred = logits.argmax(dim=1).item()
        return "background" if pred == 0 else "fire"

"""Secondary classifier loaders (YOLO or EVA02) without cross-package imports."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from timm import create_model
from torchvision import transforms
from torchvision.transforms import functional as F
from ultralytics import YOLO
from PIL import Image


class BaseClassifier:
    """Interface for classifier wrappers."""

    def predict(self, crop_image: np.ndarray) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class YOLOClassifier(BaseClassifier):
    """YOLO classification head wrapper."""

    def __init__(self, weights_path: str | Path, device: str | None = None):
        self.model = YOLO(str(weights_path))
        if device:
            self.model.to(device)

    def predict(self, crop_image: np.ndarray) -> str:
        results = self.model.predict(crop_image, verbose=False)[0]

        if results.probs is not None:
            cls_id = int(results.probs.top1)
            names = results.names
            if isinstance(names, dict):
                return names.get(cls_id, "background")
            if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
                return names[cls_id]
            return str(cls_id)

        boxes = results.boxes
        if boxes is None or boxes.data.numel() == 0:
            return "background"

        best_idx = int(boxes.conf.argmax()) if boxes.conf is not None else 0
        cls_id = int(boxes.cls[best_idx])
        names = results.names
        if isinstance(names, dict):
            return names.get(cls_id, "background")
        if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
            return names[cls_id]
        return str(cls_id)


class Letterbox:
    """Resize image with unchanged aspect ratio using padding (like YOLO)."""

    def __init__(self, size=224, color=(114, 114, 114)):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.color = color

    def __call__(self, img: Image.Image):
        w, h = img.size
        scale = min(self.size[0] / w, self.size[1] / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img_resized = F.resize(img, (new_h, new_w))
        new_img = Image.new("RGB", self.size, self.color)
        pad_left = (self.size[0] - new_w) // 2
        pad_top = (self.size[1] - new_h) // 2
        new_img.paste(img_resized, (pad_left, pad_top))
        return new_img


class CenterPadOrLetterbox:
    """If <= target size: center pad. If larger: letterbox resize."""

    def __init__(self, size=224, color=(114, 114, 114)):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.color = color

    def __call__(self, img: Image.Image):
        w, h = img.size
        if w <= self.size[0] and h <= self.size[1]:
            new_img = Image.new("RGB", self.size, self.color)
            pad_left = (self.size[0] - w) // 2
            pad_top = (self.size[1] - h) // 2
            new_img.paste(img, (pad_left, pad_top))
            return new_img
        scale = min(self.size[0] / w, self.size[1] / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img_resized = F.resize(img, (new_h, new_w))
        new_img = Image.new("RGB", self.size, self.color)
        pad_left = (self.size[0] - new_w) // 2
        pad_top = (self.size[1] - new_h) // 2
        new_img.paste(img_resized, (pad_left, pad_top))
        return new_img


class EVA02Classifier(torch.nn.Module):
    """Thin EVA02 classifier model with optional input transforms."""

    def __init__(
        self,
        model_name: str = "eva02_base_patch16_clip_224",
        num_classes: int = 2,
        pretrained: bool = True,
        transform: str | None = "centerpad",
        img_size: int = 224,
    ):
        super().__init__()
        self.backbone = create_model(model_name, pretrained=pretrained, num_classes=num_classes)

        if transform == "letterbox":
            self.transform = transforms.Compose(
                [
                    Letterbox(img_size, color=(114, 114, 114)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        elif transform == "centerpad":
            self.transform = transforms.Compose(
                [
                    CenterPadOrLetterbox(img_size, color=(114, 114, 114)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        elif transform is None:
            self.transform = None
        else:
            raise ValueError(f"Unknown transform: {transform}. Use 'letterbox', 'centerpad', or None.")

    def forward(self, x):
        return self.backbone(x)

    def get_transform(self):
        return self.transform


class EVAClassifier(BaseClassifier):
    """Wrapper around EVA02Classifier for binary fire/background style tasks."""

    def __init__(
        self,
        weights_path: str | Path,
        device: str = "cuda",
        model_name: str = "eva02_base_patch16_clip_224",
        num_classes: int = 2,
        transform: str | None = "centerpad",
        img_size: int = 224,
    ):
        self.model = EVA02Classifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False,
            transform=transform,
            img_size=img_size,
        )
        state = torch.load(weights_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.to(device).eval()
        self.device = device
        self.transform = self.model.get_transform()
        if self.transform is None:
            # Default to standard normalization if none provided
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

    def predict(self, crop_image: np.ndarray) -> str:
        if crop_image is None or crop_image.size == 0:
            return "background"

        # Convert OpenCV BGR crops to PIL RGB before applying transforms expected by the model
        if isinstance(crop_image, np.ndarray):
            if crop_image.ndim == 2:  # grayscale
                pil_image = Image.fromarray(crop_image)
            else:
                pil_image = Image.fromarray(crop_image[..., ::-1])
        else:
            pil_image = crop_image

        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            pred = logits.argmax(dim=1).item()
        return "background" if pred == 0 else "fire"


def load_classifier(
    weights: str | Path,
    model_type: Literal["yolo", "eva"] = "yolo",
    device: str | None = None,
) -> BaseClassifier:
    """Load a secondary classifier wrapper."""

    if model_type == "yolo":
        return YOLOClassifier(weights, device=device)
    if model_type == "eva":
        return EVAClassifier(weights, device=device or "cuda")
    raise ValueError(f"Unsupported classifier type: {model_type}")

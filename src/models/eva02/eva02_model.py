import torch.nn as nn
from timm import create_model
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image

class Letterbox:
    """Resize image with unchanged aspect ratio using padding (like YOLO)."""
    def __init__(self, size=224, color=(114,114,114)):
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
    def __init__(self, size=224, color=(114,114,114)):
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


class EVA02Classifier(nn.Module):
    def __init__(self, 
                 model_name="eva02_base_patch16_clip_224", 
                 num_classes=2, 
                 pretrained=True,
                 transform="letterbox",   # "letterbox" or "centerpad"
                 img_size=224):
        super().__init__()
        self.backbone = create_model(model_name, pretrained=pretrained, num_classes=num_classes)

        if transform == "letterbox":
            self.transform = transforms.Compose([
                Letterbox(img_size, color=(114,114,114)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
        elif transform == "centerpad":
            self.transform = transforms.Compose([
                CenterPadOrLetterbox(img_size, color=(114,114,114)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
        else:
            raise ValueError(f"Unknown transform: {transform}. Use 'letterbox' or 'centerpad'.")

    def forward(self, x):
        return self.backbone(x)

    def get_transform(self):
        """Expose the transform used by this model instance."""
        return self.transform

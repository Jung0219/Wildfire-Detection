import torch
import cv2
import numpy as np
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel  # required only for unpickling

IMAGE_PATH = "/lab/projects/fire_smoke_awr/data/detection/test_sets/early_fire/test/images/test/smoke_UAV000851.jpg"
YOLO_MODEL_PATH = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/early_fire_pad_aug/train/weights/best.pt"

def load_image(path, img_size=640):
    """Load image and resize with padding (letterbox)."""
    img0 = cv2.imread(path)
    assert img0 is not None, f"Image not found: {path}"
    h0, w0 = img0.shape[:2]
    scale = img_size / max(h0, w0)
    nh, nw = int(h0 * scale), int(w0 * scale)
    img_resized = cv2.resize(img0, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x = (img_size - nw) // 2
    pad_y = (img_size - nh) // 2
    img_padded = cv2.copyMakeBorder(img_resized, pad_y, img_size - nh - pad_y,
                                    pad_x, img_size - nw - pad_x,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img = img_padded[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB
    img = np.ascontiguousarray(img)
    tensor = torch.from_numpy(img).float() / 255.0
    tensor = tensor.unsqueeze(0)
    return tensor, img_padded, img0


def load_yolo_model(weights_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Safely load YOLO checkpoint (Ultralytics-style .pt file)."""
    # Allow the Ultralytics DetectionModel class to be unpickled safely
    add_safe_globals([DetectionModel])

    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    # Ultralytics checkpoints have 'model' key
    model = ckpt["model"] if "model" in ckpt else ckpt
    model = model.to(device).float()
    model.eval()
    print("YOLO model successfully loaded.")
    return model


def infer_raw(weights_path, img_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Run YOLO forward pass and get raw predictions."""
    model = load_yolo_model(weights_path, device)
    img_tensor, letterboxed, orig_img = load_image(img_path, 640)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        preds = model(img_tensor)  # forward pass (raw output)

    if isinstance(preds, (list, tuple)):
        preds = preds[0]

    print("Raw prediction tensor shape:", preds.shape)
    return preds, img_tensor, [orig_img]


if __name__ == "__main__":
    preds, img_tensor, orig_imgs = infer_raw(YOLO_MODEL_PATH, IMAGE_PATH)
    print(preds)

from typing import Any
import cv2
import numpy as np

class LetterBox:
    """
    Resize image and padding for detection, instance segmentation, pose.

    This class resizes and pads images to a specified shape while preserving aspect ratio. It also updates
    corresponding labels and bounding boxes.

    Attributes:
        new_shape (tuple): Target shape (height, width) for resizing.
        auto (bool): Whether to use minimum rectangle.
        scale_fill (bool): Whether to stretch the image to new_shape.
        scaleup (bool): Whether to allow scaling up. If False, only scale down.
        stride (int): Stride for rounding padding.
        center (bool): Whether to center the image or align to top-left.

    Methods:
        __call__: Resize and pad image, update labels and bounding boxes.

    Examples:
        >>> transform = LetterBox(new_shape=(640, 640))
        >>> result = transform(labels)
        >>> resized_img = result["img"]
        >>> updated_instances = result["instances"]
    """

    def __init__(
        self,
        new_shape: tuple[int, int] = (640, 640),
        auto: bool = False,
        scale_fill: bool = False,
        scaleup: bool = True,
        center: bool = True,
        stride: int = 32,
        padding_value: int = 114,
        interpolation: int = cv2.INTER_LINEAR,
    ):
        """
        Initialize LetterBox object for resizing and padding images.

        This class is designed to resize and pad images for object detection, instance segmentation, and pose estimation
        tasks. It supports various resizing modes including auto-sizing, scale-fill, and letterboxing.

        Args:
            new_shape (tuple[int, int]): Target size (height, width) for the resized image.
            auto (bool): If True, use minimum rectangle to resize. If False, use new_shape directly.
            scale_fill (bool): If True, stretch the image to new_shape without padding.
            scaleup (bool): If True, allow scaling up. If False, only scale down.
            center (bool): If True, center the placed image. If False, place image in top-left corner.
            stride (int): Stride of the model (e.g., 32 for YOLOv5).
            padding_value (int): Value for padding the image. Default is 114.
            interpolation (int): Interpolation method for resizing. Default is cv2.INTER_LINEAR.

        Attributes:
            new_shape (tuple[int, int]): Target size for the resized image.
            auto (bool): Flag for using minimum rectangle resizing.
            scale_fill (bool): Flag for stretching image without padding.
            scaleup (bool): Flag for allowing upscaling.
            stride (int): Stride value for ensuring image size is divisible by stride.
            padding_value (int): Value used for padding the image.
            interpolation (int): Interpolation method used for resizing.

        Examples:
            >>> letterbox = LetterBox(new_shape=(640, 640), auto=False, scale_fill=False, scaleup=True, stride=32)
            >>> resized_img = letterbox(original_img)
        """
        self.new_shape = new_shape
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left
        self.padding_value = padding_value
        self.interpolation = interpolation

    def __call__(self, labels: dict[str, Any] = None, image: np.ndarray = None) -> dict[str, Any] | np.ndarray:
        """
        Resize and pad an image for object detection, instance segmentation, or pose estimation tasks.

        This method applies letterboxing to the input image, which involves resizing the image while maintaining its
        aspect ratio and adding padding to fit the new shape. It also updates any associated labels accordingly.

        Args:
            labels (dict[str, Any] | None): A dictionary containing image data and associated labels, or empty dict if None.
            image (np.ndarray | None): The input image as a numpy array. If None, the image is taken from 'labels'.

        Returns:
            (dict[str, Any] | nd.ndarray): If 'labels' is provided, returns an updated dictionary with the resized and padded image,
                updated labels, and additional metadata. If 'labels' is empty, returns the resized
                and padded image.

        Examples:
            >>> letterbox = LetterBox(new_shape=(640, 640))
            >>> result = letterbox(labels={"img": np.zeros((480, 640, 3)), "instances": Instances(...)})
            >>> resized_img = result["img"]
            >>> updated_instances = result["instances"]
        """
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=self.interpolation)
            if img.ndim == 2:
                img = img[..., None]

        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        h, w, c = img.shape
        if c == 3:
            img = cv2.copyMakeBorder(
                img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(self.padding_value,) * 3
            )
        else:  # multispectral
            pad_img = np.full((h + top + bottom, w + left + right, c), fill_value=self.padding_value, dtype=img.dtype)
            pad_img[top : top + h, left : left + w] = img
            img = pad_img

        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, left, top)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    @staticmethod
    def _update_labels(labels: dict[str, Any], ratio: tuple[float, float], padw: float, padh: float) -> dict[str, Any]:
        """
        Update labels after applying letterboxing to an image.

        This method modifies the bounding box coordinates of instances in the labels
        to account for resizing and padding applied during letterboxing.

        Args:
            labels (dict[str, Any]): A dictionary containing image labels and instances.
            ratio (tuple[float, float]): Scaling ratios (width, height) applied to the image.
            padw (float): Padding width added to the image.
            padh (float): Padding height added to the image.

        Returns:
            (dict[str, Any]): Updated labels dictionary with modified instance coordinates.

        Examples:
            >>> letterbox = LetterBox(new_shape=(640, 640))
            >>> labels = {"instances": Instances(...)}
            >>> ratio = (0.5, 0.5)
            >>> padw, padh = 10, 20
            >>> updated_labels = letterbox._update_labels(labels, ratio, padw, padh)
        """
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels
    
    @staticmethod
    def undo_letterbox(
        boxes: np.ndarray,
        orig_shape: tuple[int, int],
        new_shape: tuple[int, int],
        scaleup: bool = True,
        center: bool = True,
        stride: int = 32,
    ) -> np.ndarray:
        """
        Undo the resizing and padding applied by LetterBox.

        This function maps bounding boxes or coordinates from a letterboxed (resized + padded)
        image back to the original image coordinate space.

        Args:
            boxes (np.ndarray): Bounding boxes in xyxy format, shape (N, 4).
            orig_shape (tuple[int, int]): Original image shape (height, width) before letterbox.
            new_shape (tuple[int, int]): Target shape used during letterboxing (height, width).
            scaleup (bool): If False, assumes the image was only scaled down (same as during letterbox).
            center (bool): Whether letterbox used centered padding.
            stride (int): Stride used during letterboxing (for auto mode).

        Returns:
            np.ndarray: Boxes mapped back to the original image coordinates.
        """
        h0, w0 = orig_shape
        h1, w1 = new_shape

        # --- Compute the same scale and padding used during LetterBox ---
        r = min(h1 / h0, w1 / w0)
        if not scaleup:
            r = min(r, 1.0)
        new_unpad = int(round(w0 * r)), int(round(h0 * r))
        dw, dh = w1 - new_unpad[0], h1 - new_unpad[1]
        if center:
            dw /= 2
            dh /= 2

        # --- Undo the letterbox ---
        boxes[:, [0, 2]] -= dw  # x padding
        boxes[:, [1, 3]] -= dh  # y padding
        boxes[:, :4] /= r  # undo scaling

        # --- Clip boxes to original image size ---
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, w0)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, h0)
        return boxes

        """
        Rescale bounding boxes from one image shape to another.

        Rescales bounding boxes from img1_shape to img0_shape, accounting for padding and aspect ratio changes.
        Supports both xyxy and xywh box formats.

        Args:
            img1_shape (tuple): Shape of the source image (height, width).
            boxes (torch.Tensor): Bounding boxes to rescale in format (N, 4).
            img0_shape (tuple): Shape of the target image (height, width).
            ratio_pad (tuple, optional): Tuple of (ratio, pad) for scaling. If None, calculated from image shapes.
            padding (bool): Whether boxes are based on YOLO-style augmented images with padding.
            xywh (bool): Whether box format is xywh (True) or xyxy (False).

        Returns:
            (torch.Tensor): Rescaled bounding boxes in the same format as input.
        """
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad_x = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
            pad_y = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)
        else:
            gain = ratio_pad[0][0]
            pad_x, pad_y = ratio_pad[1]

        if padding:
            boxes[..., 0] -= pad_x  # x padding
            boxes[..., 1] -= pad_y  # y padding
            if not xywh:
                boxes[..., 2] -= pad_x  # x padding
                boxes[..., 3] -= pad_y  # y padding
        boxes[..., :4] /= gain
        
        # ====== MANUAL CLIPPING IMPLEMENTATION ======
        h, w = img0_shape[:2]
        # assume it's a np array    
        boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], 0, w)
        boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], 0, h)
        # =============================================

        return boxes
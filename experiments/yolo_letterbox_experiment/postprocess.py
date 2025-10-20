"""
Replicating YOLO's postprocessing logic without Ultralytics dependency.
"""

import torch
import numpy as np

# ===================== NMS =====================
def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    max_det=300
):
    """Performs Non-Maximum Suppression (NMS) on inference results."""
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    output = []
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence thresholding
        if not x.shape[0]:
            output.append(torch.zeros((0, 6), device=prediction.device))
            continue

        # Compute conf = obj_conf * cls_conf
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])

        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)
        x = x[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]  # number of boxes
        if not n:
            output.append(torch.zeros((0, 6), device=prediction.device))
            continue

        c = x[:, 5:6] * (0 if agnostic else 4096)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        output.append(x[i])
    return output


# ===================== BOX OPS =====================
def xywh2xyxy(x):
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    """Rescale boxes from img1_shape to img0_shape (undo letterbox)."""
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad_x = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
        pad_y = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)
    else:
        gain = ratio_pad[0][0]
        pad_x, pad_y = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad_x
        boxes[..., 1] -= pad_y
        if not xywh:
            boxes[..., 2] -= pad_x
            boxes[..., 3] -= pad_y
    boxes[..., :4] /= gain
    return boxes if xywh else clip_boxes(boxes, img0_shape)


def clip_boxes(boxes, shape):
    """Clip bounding boxes to image boundaries."""
    h, w = shape[:2]
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, w)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, h)
    return boxes


# ===================== MAIN =====================
def postprocess(preds, img, orig_imgs, conf=0.25, iou=0.45):
    """Standalone YOLO postprocessing replication."""
    preds = non_max_suppression(preds, conf, iou)
    results = []
    for i, pred in enumerate(preds):
        if len(pred):
            pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_imgs[i].shape)
        results.append(pred.cpu().numpy())
    return results


# ===================== EXAMPLE =====================
if __name__ == "__main__":
    import torchvision
    from PIL import Image

    # Example usage: simulate raw model output
    preds = torch.rand((1, 100, 85))  # (batch, detections, [x,y,w,h,obj,cls...])
    img = torch.zeros((1, 3, 640, 640))
    orig_img = np.zeros((480, 640, 3), dtype=np.uint8)

    out = postprocess(preds, img, [orig_img])
    print("Processed boxes:", out[0].shape)

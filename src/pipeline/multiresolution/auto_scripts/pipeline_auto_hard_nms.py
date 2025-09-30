import cv2
import numpy as np
from ultralytics import YOLO
from multiresolution.composite_utils import (  # if you split into package, else just import normally
    detect_skyline_y,
    generate_composite_640x640,
    yolo_to_original,
    apply_nms,
    pad_or_downscale_to_640,
    soft_nms,
    weighted_box_fusion,
)


def process_image(img_path, model, out_txt, out_composite,
                  intermediate_size, conf_thresh=None,
                  size_thresh=None, nms_iou_thresh=None,
                  save_img=False, postproc="nms"):
    """
    Run composite detection on a single image and save results.
    postproc: one of ["nms", "softnms", "wbf", "none"]
    """
    original = cv2.imread(img_path)
    H, W = original.shape[:2]

    # Step 1: build composite
    if H >= W or H < 640 or W < 640:
        composite, (x_off, y_off, scale) = pad_or_downscale_to_640(original, 640)
        meta = {"div_y": 0, "scale_to_640": scale, "x_off": x_off, "y_off": y_off}
    else:
        y_border = detect_skyline_y(original, 120, 255, 0, 130, 5.0)
        obj_center = (0.5, float(y_border) / H) if y_border >= 0 else (0.5, 0.5)
        composite, meta = generate_composite_640x640(original, obj_center, intermediate_size)

    if save_img:
        cv2.imwrite(out_composite, composite)

    # Step 2: run YOLO
    results = model.predict(composite, imgsz=640, conf=0.001, verbose=False)[0]
    dets_top, dets_bottom = [], []

    for box, conf, cls_id in zip(results.boxes.xywhn.cpu().numpy(),
                                 results.boxes.conf.cpu().numpy(),
                                 results.boxes.cls.cpu().numpy()):
        is_bottom = (box[1] * 640 >= meta["div_y"])
        mapped = yolo_to_original(box, meta, conf, cls_id, W, H, is_bottom)
        (dets_bottom if is_bottom else dets_top).append(mapped)

    # Step 3: filter top detections
    filtered_top = []
    for det in dets_top:
        cls_id, xc, yc, w, h, conf, x1, y1, x2, y2 = det
        size = np.sqrt((x2 - x1) * (y2 - y1))
        if conf_thresh is not None and conf < conf_thresh:
            continue
        if size_thresh is not None and size >= size_thresh:
            continue
        filtered_top.append(det)

    merged = dets_bottom + filtered_top

    # Step 4: apply post-processing
    if postproc == "nms" and nms_iou_thresh and nms_iou_thresh > 0:
        final_dets = apply_nms(merged, nms_iou_thresh, W, H)
    elif postproc == "softnms":
        final_dets = soft_nms(merged, sigma=0.5, Nt=0.5, method=2)
    elif postproc == "wbf":
        final_dets = weighted_box_fusion(merged, iou_thresh=0.55)
    else:  # "none"
        final_dets = [[d[0], d[1], d[2], d[3], d[4], d[5]] for d in merged]

    # Step 5: save YOLO-format txt
    with open(out_txt, "w") as f:
        for cls_id, xc, yc, w, h, conf in final_dets:
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.4f}\n")

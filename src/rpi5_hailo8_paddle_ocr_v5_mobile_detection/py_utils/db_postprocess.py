"""DB text detector post-processing adapted from PaddleOCR (Apache-2.0)."""
import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon


def _score(heatmap, box):
    h, w = heatmap.shape
    x0, x1 = np.clip([np.floor(box[:, 0].min()), np.ceil(box[:, 0].max())], 0, w - 1).astype(int)
    y0, y1 = np.clip([np.floor(box[:, 1].min()), np.ceil(box[:, 1].max())], 0, h - 1).astype(int)
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
    shifted = box.astype(np.int32).copy(); shifted[:, 0] -= x0; shifted[:, 1] -= y0
    cv2.fillPoly(mask, [shifted], 1)
    return cv2.mean(heatmap[y0:y1 + 1, x0:x1 + 1], mask)[0]


def _unclip(box, ratio=1.5):
    polygon = Polygon(box)
    if polygon.length <= 0:
        return None
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box.astype(np.int32), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    paths = offset.Execute(polygon.area * ratio / polygon.length)
    return np.asarray(paths[0], dtype=np.float32) if len(paths) == 1 else None


def detect_boxes(output, frame, binary_threshold=0.30, box_threshold=0.60):
    heatmap = np.asarray(output).squeeze()
    if heatmap.ndim != 2:
        raise ValueError(f"Unexpected detector output shape: {np.asarray(output).shape}")
    contours, _ = cv2.findContours((heatmap > binary_threshold).astype(np.uint8) * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    src_h, src_w = frame.shape[:2]
    out_h, out_w = heatmap.shape
    boxes = []
    for contour in contours[:1000]:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        if min(rect[1]) < 3 or _score(heatmap, box) < box_threshold:
            continue
        expanded = _unclip(box)
        if expanded is None:
            continue
        rect = cv2.minAreaRect(expanded.reshape(-1, 1, 2))
        if min(rect[1]) < 5:
            continue
        box = cv2.boxPoints(rect)
        box[:, 0] = np.clip(np.round(box[:, 0] / out_w * src_w), 0, src_w - 1)
        box[:, 1] = np.clip(np.round(box[:, 1] / out_h * src_h), 0, src_h - 1)
        boxes.append(box.astype(np.int32))
    return sorted(boxes, key=lambda box: (box[:, 1].mean(), box[:, 0].mean()))

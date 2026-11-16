import numpy as np


def _nms(boxes, scores, iou_thresh, max_det=300):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-9)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep


def decode_bytetrack(hailo_output, input_h, input_w, score_thresh=0.1, nms_thresh=0.7):
    """Decode the 9 ByteTrack/YOLOX heads into person boxes.

    Output order per stride is box(4), objectness(1), class(1). The HEF already
    applies sigmoid, so scores are probabilities. Returns xyxy boxes in the
    letterboxed input space.
    """
    if hailo_output is None:
        return None, None, None
    if isinstance(hailo_output, dict):
        tensors = list(hailo_output.values())
    elif isinstance(hailo_output, (list, tuple)):
        tensors = list(hailo_output)
    else:
        tensors = [hailo_output]

    groups = {}
    for tensor in tensors:
        arr = np.asarray(tensor, dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3:
            continue
        if arr.shape[-1] not in (1, 4) and arr.shape[0] in (1, 4):
            arr = np.moveaxis(arr, 0, -1)  # CHW -> HWC
        if arr.shape[-1] not in (1, 4):
            continue
        groups.setdefault(arr.shape[:2], []).append(arr)

    all_boxes, all_scores = [], []
    for (grid_h, grid_w), group in groups.items():
        box = next((x for x in group if x.shape[-1] == 4), None)
        singles = [x for x in group if x.shape[-1] == 1]
        if box is None or len(singles) < 2:
            continue
        obj = singles[0][..., 0]
        cls = singles[1][..., 0]
        stride = int(round(input_h / grid_h))
        grid_y, grid_x = np.meshgrid(
            np.arange(grid_h, dtype=np.float32),
            np.arange(grid_w, dtype=np.float32),
            indexing="ij",
        )
        cx = (box[..., 0] + grid_x) * stride
        cy = (box[..., 1] + grid_y) * stride
        bw = np.exp(np.clip(box[..., 2], -50, 50)) * stride
        bh = np.exp(np.clip(box[..., 3], -50, 50)) * stride
        x1, y1 = cx - bw / 2, cy - bh / 2
        x2, y2 = cx + bw / 2, cy + bh / 2
        scores = obj * cls
        mask = scores >= score_thresh
        if not mask.any():
            continue
        all_boxes.append(np.stack([x1[mask], y1[mask], x2[mask], y2[mask]], axis=1))
        all_scores.append(scores[mask])

    if not all_boxes:
        return None, None, None
    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, input_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, input_h - 1)
    keep = _nms(boxes, scores, nms_thresh)
    classes = np.zeros(len(keep), dtype=np.int32)
    return boxes[keep], classes, scores[keep]
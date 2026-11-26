"""Pure post-processing helpers for the MediaPipe Face Landmarks Lite HEF.

The H8/H10 HEF exposes 1404 landmark values (468 points x 3 coordinates)
plus a single face-presence value.  The official Model Zoo post-processing
returns the first output unchanged; its coordinates are expressed in the
192x192 network input.  Older application code incorrectly treated the
model as a 98-point detector, so keep this conversion isolated and tested.
"""

import numpy as np

FACE_LANDMARK_COUNT = 468
FACE_LANDMARK_VALUES = FACE_LANDMARK_COUNT * 3
FACE_MODEL_SIZE = 192


def extract_landmarks(outputs):
    """Extract ``(points, face_score, normalized)`` from Hailo outputs.

    ``outputs`` is the dictionary returned by HailoRT.  The largest output is
    the 1404-value image-landmark tensor; a one-value output is the face
    presence score.  If the landmark coordinates are in the normalized [0, 1]
    range they are converted to 192x192 pixels; if they are already pixel
    coordinates they are left untouched.
    """
    if not outputs:
        return None, None, False

    arrays = []
    for name, value in outputs.items():
        flat = np.asarray(value, dtype=np.float32).reshape(-1)
        arrays.append((str(name), flat))

    # The landmark tensor is the largest output.  Sorting keeps this stable
    # when the HEF also exposes world landmarks of the same size.
    arrays.sort(key=lambda item: item[1].size, reverse=True)

    points = None
    face_score = None
    for _, flat in arrays:
        if points is None and flat.size >= FACE_LANDMARK_VALUES:
            points = flat[:FACE_LANDMARK_VALUES].reshape(
                FACE_LANDMARK_COUNT, 3
            ).astype(np.float32, copy=True)
        elif face_score is None and flat.size == 1:
            face_score = float(flat[0])

    if points is None:
        return None, None, False

    xy_max = float(np.max(np.abs(points[:, :2]))) if points.size else 0.0
    normalized = xy_max <= 2.0
    if normalized:
        points[:, :2] *= float(FACE_MODEL_SIZE)

    if face_score is not None and not 0.0 <= face_score <= 1.0:
        face_score = float(1.0 / (1.0 + np.exp(-face_score)))

    return points, face_score, normalized


def expand_face_box(box, frame_w, frame_h, margin=0.2):
    """Expand a YuNet box into a square crop, clamped to the frame."""
    x, y, w, h = (float(v) for v in box[:4])
    if w <= 0 or h <= 0:
        return None

    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * (1.0 + 2.0 * margin)
    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = int(round(cx + side / 2.0))
    y2 = int(round(cy + side / 2.0))

    x1 = max(0, min(frame_w - 1, x1))
    y1 = max(0, min(frame_h - 1, y1))
    x2 = max(0, min(frame_w, x2))
    y2 = max(0, min(frame_h, y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def map_landmarks_to_frame(points, crop_box):
    """Map 192x192 network landmarks back to original frame coordinates."""
    x1, y1, x2, y2 = crop_box
    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)

    mapped = np.asarray(points, dtype=np.float32).copy()
    mapped[:, 0] = x1 + (mapped[:, 0] / float(FACE_MODEL_SIZE)) * crop_w
    mapped[:, 1] = y1 + (mapped[:, 1] / float(FACE_MODEL_SIZE)) * crop_h
    return mapped
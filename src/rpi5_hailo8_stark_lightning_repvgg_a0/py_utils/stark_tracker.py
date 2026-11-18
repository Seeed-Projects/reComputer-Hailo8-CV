import cv2
import numpy as np

# Constants and crop/mapping math mirror the official Hailo Model Zoo
# `tracker_infer.py` for stark_lightning_repvgg_a0 (MIT-licensed STARK
# reference implementation).
TEMPLATE_FACTOR = 2.0
TEMPLATE_SIZE = 128
SEARCH_FACTOR = 5.0
SEARCH_SIZE = 320
CLIP_MARGIN = 10.0


def sample_target(image, box, area_factor, output_sz):
    """Crop a square region around `box`, resize it, and return the scale.

    Args:
        image: HxWx3 BGR frame.
        box: (x, y, w, h) in image pixels.
        area_factor: template 2.0, search 5.0.
        output_sz: 128 (template) or 320 (search).

    Returns:
        (crop, resize_factor) where resize_factor = output_sz / crop_sz.
    """
    x, y, w, h = [float(v) for v in box]
    crop_sz = int(np.ceil(np.sqrt(max(w, 1e-3) * max(h, 1e-3)) * area_factor))
    crop_sz = max(crop_sz, 1)
    x1 = int(round(x + 0.5 * w - crop_sz * 0.5))
    y1 = int(round(y + 0.5 * h - crop_sz * 0.5))
    x2, y2 = x1 + crop_sz, y1 + crop_sz

    x1_pad, y1_pad = max(0, -x1), max(0, -y1)
    x2_pad = max(x2 - image.shape[1] + 1, 0)
    y2_pad = max(y2 - image.shape[0] + 1, 0)

    patch = image[
        y1 + y1_pad : y2 - y2_pad,
        x1 + x1_pad : x2 - x2_pad,
        :,
    ]
    patch = np.pad(
        patch,
        ((y1_pad, y2_pad), (x1_pad, x2_pad), (0, 0)),
        mode="constant",
    )
    resized = cv2.resize(
        patch, (output_sz, output_sz), interpolation=cv2.INTER_LINEAR
    )
    return resized, output_sz / crop_sz


def clip_box(box, height, width, margin=0):
    """Clamp an xywh box into the frame, matching STARK's clip_box."""
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), width - margin)
    x2 = min(max(margin, x2), width)
    y1 = min(max(0, y1), height - margin)
    y2 = min(max(margin, y2), height)
    return [x1, y1, max(margin, x2 - x1), max(margin, y2 - y1)]


def map_box_back(box_norm, state, resize_factor, height, width):
    """Map the head's normalized xyxy from the search crop to image xywh."""
    box_px = np.asarray(box_norm, dtype=np.float64) * SEARCH_SIZE / resize_factor
    cx_prev = state[0] + 0.5 * state[2]
    cy_prev = state[1] + 0.5 * state[3]
    half = 0.5 * SEARCH_SIZE / resize_factor
    x1 = box_px[0] + (cx_prev - half)
    y1 = box_px[1] + (cy_prev - half)
    x2 = box_px[2] + (cx_prev - half)
    y2 = box_px[3] + (cy_prev - half)
    clipped = clip_box(
        [x1, y1, x2 - x1, y2 - y1], height, width, margin=CLIP_MARGIN
    )
    return np.asarray(clipped, dtype=np.float64)


class StarkTracker:
    """Stateful one-shot tracker around the STARK HEF."""

    def __init__(self, model):
        self.model = model
        self.template = None
        self.state = None

    @property
    def initialized(self):
        return self.template is not None and self.state is not None

    def initialize(self, frame, box):
        """Seed the template from the first frame and the target box."""
        x, y, w, h = [float(v) for v in box]
        if w <= 0 or h <= 0:
            raise ValueError("Initial box must have positive width and height")
        self.state = np.asarray([x, y, w, h], dtype=np.float64)
        template, _ = sample_target(
            frame, self.state, TEMPLATE_FACTOR, TEMPLATE_SIZE
        )
        self.template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
        return self.state.copy()

    def update(self, frame):
        """Track one frame and return the new xywh state in image pixels."""
        if not self.initialized:
            raise RuntimeError("Call initialize() before update()")

        search, resize_factor = sample_target(
            frame, self.state, SEARCH_FACTOR, SEARCH_SIZE
        )
        search = cv2.cvtColor(search, cv2.COLOR_BGR2RGB)

        outputs = self.model.run(self.template, search)
        if outputs is None:
            return self.state.copy()

        box_norm = np.asarray(list(outputs.values())[0], dtype=np.float64).reshape(-1)[:4]
        if box_norm.size != 4 or not np.all(np.isfinite(box_norm)):
            return self.state.copy()

        self.state = map_box_back(
            box_norm,
            self.state,
            resize_factor,
            frame.shape[0],
            frame.shape[1],
        )
        return self.state.copy()

    def reset(self):
        self.template = None
        self.state = None
# Test Report - YOLOv8m-seg on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | yolov8m_seg |
| Task | Instance segmentation (COCO 80 classes, one2many head) |
| Backbone | YOLOv8m-seg (Ultralytics, AGPL-3.0) |
| Parameters | 27.3M |
| Operations | 110.2G |
| Family | yolov8_seg (hailo8: n/s/m; hailo10h: n/s/m) |

## HEF artifact

| Field | Value |
|---|---|
| File | model/yolov8m_seg.hef |
| Size | 32,930,063 bytes |
| SHA256 | `f77dd2985a7614d39643076a98fb69e885a280f90435e2e7ca7eedcf10ad78dd` |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/yolov8m_seg.hef |

## Input / output

| Field | Value |
|---|---|
| Input | 640x640x3 RGB uint8 (letterbox pad 114) |
| Output | 10 raw heads: 3 strides x {DFL box 64ch, score 80ch, mask 32ch} + proto 160x160x32 |
| Decode | DFL (16 bins) + sigmoid + per-class NMS (100 max detections in the preview) |
| Masks | sigmoid(coeffs @ proto) cropped per box (threshold 0.5) |

## Verification status

| Check | Status |
|---|---|
| Python syntax (py_compile) | Pass |
| Offline decode harness (planted synthetic heads -> exact box, DFL and folded layouts) | Pass |
| Preview and offline-analysis box mapping (unletterbox_boxes before draw_boxes, same form as the Hailo-10H modules) | Pass |
| CI matrix entry added | Pass |

### Hardware verification - pending (assigned to the maintainer)

| Check | Status |
|---|---|
| HEF loads, 10 output vstreams match the documented shapes | Pending |
| Demo video: instance masks + boxes align | Pending |
| Single-image REST API | Pending |
| Offline video analysis | Pending |
| USB camera | Pending |
| Official GHCR image re-pull + run | Pending |
| FPS measurement (CPU decode is the bottleneck) | Pending |

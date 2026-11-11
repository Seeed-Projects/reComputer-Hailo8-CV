# Test Report - YOLO11x-seg on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | yolov11x_seg |
| Task | Instance segmentation (COCO 80 classes, one2many head) |
| Backbone | YOLO11x-seg (Ultralytics, AGPL-3.0) |
| Parameters | 62.1M |
| Operations | 319.4G |
| mAP-seg (float / Hailo-8) | 43.8 / 43.4 |
| Family | yolov11_seg (hailo8: n/s/m/l/x) |

## HEF artifact

| Field | Value |
|---|---|
| File | model/yolov11x_seg.hef |
| Size | 68,013,298 bytes |
| SHA256 | `a55b8dc835d17d5187a92a47be8cd66d64b2ca503302d36fca24e64f7b85cc38` |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/yolov11x_seg.hef |

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
| Offline decode harness (synthetic 10-head payload -> box at the letterboxed frame position) | Pass |
| Preview box mapping (unletterbox_boxes before draw_boxes) | Pass |
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

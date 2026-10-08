# Test Report - YOLOv8s-seg on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | yolov8s_seg |
| Task | Instance segmentation (COCO 80 classes, one2many head) |
| Backbone | YOLOv8s-seg (Ultralytics, AGPL-3.0) |
| Parameters | 11.8M |
| Operations | 42.6G |
| Family | yolov8_seg (hailo8: n/s/m; hailo10h: n/s/m) |

## HEF artifact

| Field | Value |
|---|---|
| File | model/yolov8s_seg.hef |
| Size | 17,339,012 bytes |
| SHA256 | `eeb9a80b0db82c32bffc9a581d1f4930bed04a0499d9c2112727c4d525da562a` |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/yolov8s_seg.hef |

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

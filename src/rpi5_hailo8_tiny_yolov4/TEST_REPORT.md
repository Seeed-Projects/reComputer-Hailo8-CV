# Test Report — Tiny-YOLOv4 on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | tiny_yolov4 |
| Task | Object detection (COCO 80 classes) |
| Backbone | YOLOv4-tiny (base/yolo.yaml) |
| Parameters | 6.05M |
| Operations | 6.92G |
| Framework source | pytorch (Tianxiaomo/pytorch-YOLOv4) |
| Model Zoo version | v2.19.0 |
| Compile target | hailo8 |
| Family | yolo (variants: tiny_yolov3 / tiny_yolov4) |
| Sibling | tiny_yolov3 (same I/O contract, YOLOv3-tiny backbone) |

## HEF artifact

| Field | Value |
|---|---|
| File | model/tiny_yolov4.hef |
| Size | ~7.3 MB (7,589,424 bytes) |
| SHA256 | 7260b71cdbde9533b50da2f0fab30b7034a60c68e3ef4dd5b152030c8133bd28 |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/tiny_yolov4.hef |

## Input / output (same as tiny_yolov3)

| Field | Value |
|---|---|
| Input | 416x416x3 RGB, uint8 (normalize_in_net std=255) |
| Padding | color 114 (gray) |
| On-chip NMS | NO (raw heads, CPU decode) |
| Output | 13x13x255 (stride 32) + 26x26x255 (stride 16) |
| Anchors | Same as tiny_yolov3: [[81,82],[135,169],[344,319]] / [[23,27],[37,58],[81,82]] |
| num_classes | 80 (COCO, 0-indexed) |

## Verification status

| Check | Status |
|---|---|
| Python syntax (py_compile) | Pass |
| Module structure matches SOP | Pass |
| Dockerfile present + CMD references real HEF | Pass |
| CI matrix entry added | Pass |
| Post-processing identical to verified tiny_yolov3 (CPU YOLOv3 decode) | Pass |

### Hardware verification — pending

| Check | Status |
|---|---|
| HEF loads | Pending |
| Head shapes confirmed (13x13x255, 26x26x255) | Pending |
| Anchor decode correct | Pending |
| Demo video: boxes align | Pending |
| Official GHCR image re-pull + run | Pending |

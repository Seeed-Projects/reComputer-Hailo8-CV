# Test Report — YOLO26n on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | yolo26n |
| Task | Object detection (COCO 80 classes) |
| Backbone | YOLO26n (ultralytics, one2one head) |
| Parameters | 2.4M |
| Operations | 5.5G |
| Framework source | pytorch (ultralytics/ultralytics) |
| Model Zoo version | v2.19.0 |
| Compile target | Hailo-8 |
| Family | yolo26 (detection; n/s/m are published for both Hailo-8 and Hailo-10H) |

## HEF artifact

| Field | Value |
|---|---|
| File | model/yolo26n.hef |
| Size | 8,755,512 bytes |
| SHA256 | 743de04972fbdba05140d18b9176e38620011134e9f79c7a7dd63b1d13eb90ae |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/yolo26n.hef |

## Input / output

| Field | Value |
|---|---|
| Input | 640x640x3 RGB, uint8 (normalize_in_net mean 0 / std 255) |
| Padding | color 114 (gray) |
| Zoo postprocessing | nms=false, sigmoid=false, hpp=false, meta_arch=yolo26 |
| Zoo output_shape | 3 x (4 box, 80 class scores) — raw split heads |
| Output actually returned | to be confirmed on hardware (first-inference log) |
| Selection | ultralytics two-stage top-k, post_nms_topk=300, no NMS |
| num_classes | 80 (COCO, 0-indexed) |

## Logic checks (synthetic tensors, no device)

| Check | Result |
|---|---|
| Raw 4-channel box heads decode (l, t / r, b around the cell centre) | Pass |
| Raw 64-channel DFL heads decode through the softmax expectation | Pass |
| One2one top-k returns the hot (anchor, class) pair, no NMS | Pass |
| Confidence threshold drops the detection | Pass |
| On-chip NMS payload (ragged NMS-by-score list, dense and compact) parses | Pass |
| Unexpected layout returns no detections instead of raising | Pass |
| Empty payload returns no detections | Pass |

## Verification status

| Check | Status |
|---|---|
| Python syntax (py_compile) | Pass |
| Module structure matches SOP | Pass |
| Dockerfile present + CMD references the real HEF | Pass |
| CI matrix entry added | Pass |
| Demo video on CM5 + Hailo-8 | **Not executed** — no device in this environment |
| Single-image REST API on device | **Not executed** |
| Offline video analysis on device | **Not executed** |
| USB camera on device | **Not executed** |

## Known issues

- The Model Zoo documents the raw split heads while a compile may fold on-chip
  NMS in; the module handles both and logs which one the HEF returned on the
  first inference. The device run must confirm the layout and value ranges.
- YOLO26l / YOLO26x have no Hailo-8 or Hailo-10H HEFs in the Model Zoo yet.

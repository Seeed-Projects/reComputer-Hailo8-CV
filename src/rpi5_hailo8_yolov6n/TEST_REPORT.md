# Test Report — YOLOv6n on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | yolov6n |
| Task | Object detection (COCO 80 classes) |
| Backbone | YOLOv6n (Meituan) |
| Parameters | 4.32M |
| Operations | 11.12G |
| Framework source | pytorch (meituan/YOLOv6) |
| Model Zoo version | v2.19.0 |
| Compile target | hailo8 |
| Family | yolov6 (only yolov6n is compiled in the Model Zoo; s/m/l are not) |

## HEF artifact

| Field | Value |
|---|---|
| File | model/yolov6n.hef |
| Size | 5,774,047 bytes |
| SHA256 | ede85fec687252a7963f6bafbc8e9cefbadf27e3b58c5133ead50f2c8c5a8ee7 |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/yolov6n.hef |

## Input / output

| Field | Value |
|---|---|
| Input | 640x640x3 RGB, uint8 (normalize_in_net mean 0 / std 255) |
| Padding | color 114 (gray) |
| Zoo postprocessing | hpp=true, meta_arch=yolo_v6, score_threshold=0.03, nms_iou_thresh=0.65 |
| Zoo output_shape | 3 x (4 box, 1 objectness, 80 classes) — raw split heads |
| Output actually returned | to be confirmed on hardware (first-inference log) |
| num_classes | 80 (COCO, 0-indexed; labels_offset=1 for category ids) |

## Logic checks (synthetic tensors, no device)

| Check | Result |
|---|---|
| On-chip NMS compact buffer parses to the expected box/class/score | Pass |
| Dense Cx5xD and CxDx5 layouts parse | Pass |
| Ragged NMS-by-score per-class list parses | Pass |
| Raw split heads decode (l,t / r,b around the cell centre, stride units) | Pass |
| objectness x class score and per-class NMS keep one box for a duplicate | Pass |
| Confidence threshold drops the detection | Pass |
| Unexpected layout returns no detections instead of raising | Pass |

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

- The Model Zoo documents the raw split heads while `hpp: true` may fold NMS
  into the compile; the module handles both and logs which one the HEF returned
  on the first inference. The device run must confirm the layout and the value
  ranges.
- Only `yolov6n` has Hailo-8/Hailo-10H HEFs in the Model Zoo; `yolov6s/m/l`
  cannot be added until Hailo publishes them.

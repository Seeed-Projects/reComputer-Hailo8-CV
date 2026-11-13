# Test Report - YOLOv3-Gluon-416 on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | yolov3_gluon_416 |
| Task | Object detection (COCO 80 classes) |
| Framework source | GluonCV |
| Parameters | 61.92M |
| Operations | 65.94G |
| mAP (float / Hailo-8) | 36.3 / 34.3 |
| Model Zoo version | v2.20.0 |
| Compile target | hailo8 |

## HEF artifact

| Field | Value |
|---|---|
| File | model/yolov3_gluon_416.hef |
| Size | 56,194,011 bytes |
| SHA256 | `b9973536196090ac5724fe3e875c841cfae47ad6540d44ae04122bdbef1281f9` |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/yolov3_gluon_416.hef |

## Input / output

| Field | Value |
|---|---|
| Input | 416x416x3 RGB uint8 (letterbox, padding 114) |
| On-chip NMS | NO (raw heads, CPU decode) |
| Output | 3 tensors: 13x13 / 26x26 / 52x52, 255 channels each |
| Per-head layout | 255 = 3 anchors x (4 box + 1 obj + 80 cls) |
| Anchors stride 32 | [[116,90],[156,198],[373,326]] |
| Anchors stride 16 | [[30,61],[62,45],[59,119]] |
| Anchors stride 8 | [[10,13],[16,30],[33,23]] |
| num_classes | 80 (COCO, 0-indexed) |
| score threshold (eval) | 0.1 |
| nms_iou_thresh (eval) | 0.3 |

## Verification status

| Check | Status |
|---|---|
| Python syntax (py_compile) | Pass |
| Offline three-head decode harness | Pass |
| Configuration defaults (confidence 0.25, NMS 0.30) | Pass |
| Dockerfile present + CMD references real HEF | Pass |
| CI matrix entry added | Pass |
| HEF input/output taken from Model Zoo source (not guessed) | Pass |
| First-inference vstream logging implemented (SOP §10) | Pass |

### Hardware verification - pending

| Check | Status |
|---|---|
| HEF loads, no driver/firmware mismatch | Pending |
| Output head shapes confirmed (13x13 / 26x26 / 52x52) | Pending |
| Anchor decode correct (boxes align with objects) | Pending |
| Demo video: boxes align, correct classes | Pending |
| Single-image REST API | Pending |
| Official GHCR image re-pull + run | Pending |

## Known points to verify on hardware

- The head-to-stride mapping is by spatial size (ascending area = stride 32, 16, 8).
  Verify the first-inference log shows the expected three grids.
- Anchors are in pixel units. If boxes look wrong, verify anchor order or the
  CHW→HWC transpose on hardware.

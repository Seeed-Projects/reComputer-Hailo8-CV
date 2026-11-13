# Test Report - YOLOv3 on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | yolov3 |
| Task | Object detection (COCO 80 classes) |
| Framework source | Darknet (AlexeyAB) |
| Parameters | 68.79M |
| Operations | 158.10G |
| mAP (float / Hailo-8) | 38.4 / 38.3 |
| Model Zoo version | v2.20.0 |
| Compile target | hailo8 |

## HEF artifact

| Field | Value |
|---|---|
| File | model/yolov3.hef |
| Size | 54,686,971 bytes |
| SHA256 | `8a525ff5b77d03e812b33b2d7828b7c983d2546c2043fc03dab39d9bc8f2d9ab` |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/yolov3.hef |

## Input / output

| Field | Value |
|---|---|
| Input | 608x608x3 RGB uint8 (letterbox, padding 114) |
| On-chip NMS | NO (raw heads, CPU decode) |
| Output | 3 tensors: 19x19 / 38x38 / 76x76, 255 channels each |
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
| Output head shapes confirmed (19x19 / 38x38 / 76x76) | Pending |
| Anchor decode correct (boxes align with objects) | Pending |
| Demo video: boxes align, correct classes | Pending |
| Single-image REST API | Pending |
| Official GHCR image re-pull + run | Pending |

## Known points to verify on hardware

- The head-to-stride mapping is by spatial size (ascending area = stride 32, 16, 8).
  Verify the first-inference log shows the expected three grids.
- Anchors are in pixel units. If boxes look wrong, verify anchor order or the
  CHW→HWC transpose on hardware.

# Test Report — Tiny-YOLOv3 on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | tiny_yolov3 |
| Task | Object detection (COCO 80 classes) |
| Backbone | Tiny-YOLOv3 (base/yolo.yaml) |
| Parameters | 8.85M |
| Operations | 5.58G |
| Framework source | pytorch (Tianxiaomo/pytorch-YOLOv4) |
| Model Zoo version | v2.19.0 |
| Compile target | hailo8 |
| Family | yolo (variants: tiny_yolov3 / tiny_yolov4 planned) |

## HEF artifact

| Field | Value |
|---|---|
| File | model/tiny_yolov3.hef |
| Size | ~8.4 MB (8,722,414 bytes) |
| SHA256 | 4af8a7beecac4ef0a3f1dce58c2a8813a19c107b2b4c9c2e40a94827bf9d6780 |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/tiny_yolov3.hef |

## Input / output (from Model Zoo network YAML + base/yolo.yaml)

| Field | Value |
|---|---|
| Input | 416x416x3 RGB, uint8 (normalize_in_net std=255, ÷255) |
| Padding | color 114 (gray, YOLO convention) |
| On-chip NMS | NO (raw heads, CPU decode) |
| Output | 2 tensors: 13x13x255 (stride 32) + 26x26x255 (stride 16) |
| Per-head layout | 255 = 3 anchors × (4 box + 1 obj + 80 cls) |
| Anchors stride 32 | [[81,82],[135,169],[344,319]] |
| Anchors stride 16 | [[23,27],[37,58],[81,82]] |
| num_classes | 80 (COCO, 0-indexed) |
| score_threshold (eval) | 0.1 |
| nms_iou_thresh (eval) | 0.3 |

## Verification status

| Check | Status |
|---|---|
| Python syntax (py_compile) | Pass |
| Module structure matches SOP | Pass |
| Dockerfile present + CMD references real HEF | Pass |
| CI matrix entry added | Pass |
| HEF input/output taken from Model Zoo source (not guessed) | Pass |
| First-inference vstream logging implemented (SOP §10) | Pass |
| Post-processing ported from official yolo.py _yolo3_decode (CPU decode) | Pass |

### Hardware verification — pending

| Check | Status |
|---|---|
| HEF loads, no driver/firmware mismatch | Pending |
| Output head shapes confirmed (13x13x255, 26x26x255) | Pending |
| Anchor decode correct (boxes align with objects) | Pending |
| Demo video: boxes align, correct classes | Pending |
| Single-image REST API: class/confidence/box correct | Pending |
| Official GHCR image re-pull + run | Pending |

## Known points to verify on hardware

- This is the **first CPU-decode model** in the repo (all others use on-chip
  NMS). The YOLOv3 grid decode + sigmoid + NMS run entirely on the CPU — verify
  the decode is correct on hardware (boxes align, correct classes).
- The head-to-stride mapping is by spatial size (smaller grid = larger stride).
  Verify the first-inference log shows 13x13 + 26x26 heads.
- The anchors are from the network YAML (pixel units). If boxes look wrong
  (too large/small), the anchor order or the CHW→HWC transpose may need
  adjustment — verify on hardware.

# Test Report — YOLOX-Tiny on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | yolox_tiny |
| Task | Object detection (COCO 80 classes) |
| Backbone | YOLOX-Tiny (Megvii) |
| Parameters | 5.05M |
| Operations | 6.44G |
| Framework source | pytorch (Megvii-BaseDetection/YOLOX) |
| Model Zoo version | v2.19.0 |
| Compile target | hailo8 |
| Family | yolox (variants: tiny / s_leaky / l_leaky planned) |

## HEF artifact

| Field | Value |
|---|---|
| File | model/yolox_tiny.hef |
| Size | ~5.5 MB |
| SHA256 | 5d6d92d5ebca0b94cb840ba69d71f7fdb9eaf5d768af96b759a1650831ef337d |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/yolox_tiny.hef |

## Input / output

| Field | Value |
|---|---|
| Input | 416x416x3 RGB, uint8 (normalize_in_net ImageNet RGB) |
| Padding | color 114 (gray, YOLOX convention) |
| On-chip | nms=true, hpp=true, meta_arch=yolox |
| Output | 80x5x100 (post-NMS HPP) |
| num_classes | 80 (COCO, 0-indexed) |
| score_threshold (eval) | 0.01 |
| nms_iou_thresh (eval) | 0.65 |

## Verification status

| Check | Status |
|---|---|
| Python syntax (py_compile) | Pass |
| Module structure matches SOP | Pass |
| Dockerfile present + CMD references real HEF | Pass |
| CI matrix entry added | Pass |
| Post-processing (HPP ragged-safe, 80-class) | Pass |

### Hardware verification — pending

| Check | Status |
|---|---|
| HEF loads | Pending |
| Output shape confirmed (80x5x100) | Pending |
| Ragged NMS-by-score layout confirmed | Pending |
| Class mapping correct (80-class, 0-indexed) | Pending |
| Demo video: boxes align | Pending |
| Official GHCR image re-pull + run | Pending |

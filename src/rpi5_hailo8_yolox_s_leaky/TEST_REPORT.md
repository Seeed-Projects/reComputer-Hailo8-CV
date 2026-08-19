# Test Report — YOLOX-S-Leaky on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | yolox_s_leaky |
| Task | Object detection (COCO 80 classes) |
| Backbone | YOLOX-S-Leaky (Megvii) |
| Framework source | pytorch (Megvii-BaseDetection/YOLOX) |
| Model Zoo version | v2.19.0 |
| Compile target | hailo8 |
| Family | yolox (variants: tiny / s_leaky / l_leaky) |
| Sibling | yolox_tiny (same I/O, 416 input, lighter) |

## HEF artifact

| Field | Value |
|---|---|
| File | model/yolox_s_leaky.hef |
| Size | ~9 MB (9,383,447 bytes) |
| SHA256 | f51c6c2c6cd1bd73b9171858809177b39230ef98e6ff2de7306d313b013cf9be |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/yolox_s_leaky.hef |

## Input / output (same as yolox_tiny, 640 input)

| Field | Value |
|---|---|
| Input | 640x640x3 RGB, uint8 (normalize_in_net ImageNet RGB) |
| Padding | color 114 (gray, YOLOX convention) |
| On-chip | nms=true, hpp=true, meta_arch=yolox |
| Output | 80x5x100 (post-NMS HPP) |
| num_classes | 80 (COCO, 0-indexed) |

## Verification status

| Check | Status |
|---|---|
| Python syntax (py_compile) | Pass |
| Module structure matches SOP | Pass |
| Dockerfile present + CMD references real HEF | Pass |
| CI matrix entry added | Pass |

### Hardware verification — pending

| Check | Status |
|---|---|
| HEF loads | Pending |
| Output shape confirmed (80x5x100) | Pending |
| Demo video: boxes align | Pending |
| Official GHCR image re-pull + run | Pending |
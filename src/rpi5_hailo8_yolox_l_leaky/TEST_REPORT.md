# Test Report — YOLOX-L-Leaky on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | yolox_l_leaky |
| Task | Object detection (COCO 80 classes) |
| Backbone | YOLOX-L-Leaky (Megvii) |
| Parameters | 54.17M |
| Operations | 155.3G |
| Family | yolox (tiny / s_leaky / l_leaky) |

## HEF artifact

| Field | Value |
|---|---|
| File | model/yolox_l_leaky.hef |
| Size | ~53 MB |
| SHA256 | 735098b1e7223d06343425e090842a3dbac5d2b42694f72d4978092c41f8354d |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/yolox_l_leaky.hef |

## Input / output (same as s_leaky)

| Field | Value |
|---|---|
| Input | 640x640x3 RGB, uint8 |
| Output | 80x5x100 (post-NMS HPP) |
| num_classes | 80 |

## Verification status

| Check | Status |
|---|---|
| Python syntax (py_compile) | Pass |
| CI matrix entry added | Pass |

### Hardware verification — pending

| Check | Status |
|---|---|
| HEF loads | Pending |
| Demo video: boxes align | Pending |
| Official GHCR image re-pull + run | Pending |
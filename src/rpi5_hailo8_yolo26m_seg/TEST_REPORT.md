# Test Report — YOLO26m-seg on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | yolo26m_seg |
| Task | Instance segmentation (COCO 80 classes, one2one heads) |
| Backbone | YOLO26m-seg (Ultralytics, AGPL-3.0) |
| Parameters | 23.6M |
| Operations | ~68G |
| Family | yolo26_seg (hailo8: m; hailo10h: n/s/m) |

## HEF artifact

| Field | Value |
|---|---|
| File | model/yolo26m_seg.hef |
| Size | 33,718,605 bytes (~33.7 MB) |
| SHA256 | 2e1b7bd911cd1ec725b9fa930eb067b6b9b3d429a98ba2ac5db210faaac74e0f |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/yolo26m_seg.hef |

Note: the Hailo-8 Model Zoo only ships compiled builds of yolo26m_seg and
yolo26x_seg — n/s variants do not exist for this architecture.

## Input / output

| Field | Value |
|---|---|
| Input | 640x640x3 RGB uint8 (letterbox pad 0) |
| Output | 10 heads: 3 strides x {bbox 64ch, score 80ch, mask 32ch} + proto 160x160x32 |
| Decode | one2one DFL (16 bins) + two-stage top-k (no NMS) |
| Masks | sigmoid(coeffs @ proto) cropped per box |

## Verification status

| Check | Status |
|---|---|
| Python syntax (py_compile) | Pass |
| Offline decode test (planted cell -> exact box) | Pass (ported from validated 10H module) |
| CI matrix entry added | Pass |

### Hardware verification — pending

| Check | Status |
|---|---|
| HEF loads, 10 output vstreams match YAML shapes | Pending |
| Demo video: instance masks + boxes align | Pending |
| Official GHCR image re-pull + run | Pending |
| FPS measurement (CPU decode is the bottleneck) | Pending |
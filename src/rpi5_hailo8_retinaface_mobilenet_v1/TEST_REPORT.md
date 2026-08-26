# Test Report — RetinaFace MobileNet-v1 on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | retinaface_mobilenet_v1 |
| Task | Face detection (single class + 5 landmarks) |
| Backbone | MobileNet-v1 (RetinaFace, biubug6/Pytorch_Retinaface) |
| Parameters | 3.49M |
| Operations | 25.14G |
| Family | retinaface (mobilenet_v1) |

## HEF artifact

| Field | Value |
|---|---|
| File | model/retinaface_mobilenet_v1.hef |
| Size | ~6.3 MB |
| SHA256 | 792c618c489bf966e779b31d152933f2c24eca0b5af07b5ff9402d78870f5834 |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/retinaface_mobilenet_v1.hef |

## Input / output

| Field | Value |
|---|---|
| Input | 736x1280x3 BGR uint8 (normalize_in_net mean [123,117,104], std 1) |
| Output | 9 heads: 3 scales x {bbox 8ch, conf 4ch, landmark 20ch} |
| Priors | 38,640 (2/cell, feature maps 92x160 / 46x80 / 23x40) |
| Decode | SSD variances (10, 5); conf softmax slice(1) |
| NMS | greedy, iou 0.4, score 0.02 (official eval config) |
| Landmarks | 5 keypoints (eyes, nose, mouth corners) per face |

## Verification status

| Check | Status |
|---|---|
| Python syntax (py_compile) | Pass |
| Offline pipeline smoke test (synthetic 9 heads) | Pass |
| CI matrix entry added | Pass |

### Hardware verification — pending

| Check | Status |
|---|---|
| HEF loads and 9 output vstreams match YAML shapes | Pending |
| Demo video: faces + 5 keypoints align | Pending |
| Official GHCR image re-pull + run | Pending |
| FPS measurement | Pending |
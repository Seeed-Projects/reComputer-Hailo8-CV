# Test Report — YOLACT-RegNetX-1.6GF on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | yolact_regnetx_1_6gf |
| Task | Instance segmentation (COCO 80 classes) |
| Backbone | RegNetX-1.6GF (YOLACT, UC Davis) |
| Parameters | 30.09M |
| Operations | 125.34G |
| Family | yolact (regnetx_800mf / regnetx_1.6gf) |

## HEF artifact

| Field | Value |
|---|---|
| File | model/yolact_regnetx_1_6gf.hef |
| Size | ~35 MB |
| SHA256 | e0a8635e080be28804e2f60272db2d6a85223a2651dce55af8e44c2ee9158ac9 |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/yolact_regnetx_1_6gf.hef |

## Input / output

| Field | Value |
|---|---|
| Input | 512x512x3 RGB, uint8 (normalize_in_net ImageNet mean/std) |
| Output | 16 heads: proto (128x128x32), 5x {bbox 36, mask 288, conf 729} |
| Priors | 49,104 (9/cell, feature maps 64/32/16/8/4) |
| Fast NMS | top_k=200/class, iou=0.5, max 100 dets |
| Classes | 80 (COCO, Model Zoo index order) |

## Verification status

| Check | Status |
|---|---|
| Python syntax (py_compile) | Pass |
| Offline pipeline smoke test (synthetic 16 heads) | Pass |
| CI matrix entry added | Pass |

### Hardware verification — pending

| Check | Status |
|---|---|
| HEF loads and 16 output vstreams match YAML shapes | Pending |
| Demo video: masks + boxes align on instances | Pending |
| Official GHCR image re-pull + run | Pending |
| FPS measurement (CPU post-process is the bottleneck) | Pending |
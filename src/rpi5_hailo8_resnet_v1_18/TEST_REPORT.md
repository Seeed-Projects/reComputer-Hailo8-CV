# Test Report - ResNet-18 on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | resnet_v1_18 |
| Task | ImageNet-1k classification |
| Backbone | ResNet-18 (BRECQ) |
| Parameters | 11.68M |
| Operations | 3.64G |
| Top-1 accuracy (float) | 71.26% |
| Model Zoo version | v2.20.0 |
| Compile target | hailo8 |

## HEF artifact

| Field | Value |
|---|---|
| File | model/resnet_v1_18.hef |
| Size | 12,463,716 bytes |
| SHA256 | `9fba94d31cdde220857e05c1b101de8d9044e53b107ef5b324ed5863fae8d965` |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/resnet_v1_18.hef |

## Input / output

| Field | Value |
|---|---|
| Input | 224x224x3 RGB uint8 (plain resize; ImageNet normalization compiled in the HEF) |
| Output | 1000 ImageNet-1k scores (softmax disabled on device) |
| Decode | host softmax + Top-K, EMA smoothing for the preview |
| Overlay | Top-K panel (label, percentage, bar) in the top-left corner |

## Demo video

`video/test.mp4` is Hailo's official classifier demo asset `example.mp4`
(1280x720, 11.2 s, street scene) from
`https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/video/example.mp4`
(`hailo-ai/hailo-apps`, MIT).

## Verification status

| Check | Status |
|---|---|
| Python syntax (py_compile) | Pass |
| Offline inference_loop harness (synthetic 1000-way logits -> Top-K overlay) | Pass |
| Configuration defaults (confidence 0.30, Top-K 5) | Pass |
| Dockerfile present + CMD references real HEF | Pass |
| CI matrix entry added | Pass |

### Hardware verification - pending (assigned to the maintainer)

| Check | Status |
|---|---|
| HEF loads, output is a 1000-way vector | Pending |
| Label mapping matches ImageNet order | Pending |
| Demo video: Top-K labels are stable and plausible | Pending |
| Single-image REST API | Pending |
| Offline video analysis | Pending |
| USB camera | Pending |
| Official GHCR image re-pull + run | Pending |
| FPS measurement | Pending |
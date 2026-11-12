# Test Report - ResNet-50 on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | resnet_v1_50 |
| Task | ImageNet-1k classification |
| Backbone | ResNet-50 (BRECQ) |
| Parameters | 25.53M |
| Operations | 6.98G |
| Top-1 accuracy (float / Hailo-8) | 75.2% / 74.7% |
| Model Zoo version | v2.20.0 |
| Compile target | hailo8 |

## HEF artifact

| Field | Value |
|---|---|
| File | model/resnet_v1_50.hef |
| Size | 17,978,280 bytes |
| SHA256 | `6dc70520ce6306ff1082041d1e796ff3165813e1e003603019c9e27cc11de55c` |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/resnet_v1_50.hef |

## Input / output

| Field | Value |
|---|---|
| Input | 224x224x3 RGB uint8 (plain resize; ImageNet normalization compiled in the HEF) |
| Output | 1000 ImageNet-1k scores (softmax disabled on device) |
| Decode | host softmax + Top-K, EMA smoothing for the preview |
| Overlay | Top-K panel (label, percentage, bar) in the top-left corner |

## Demo video

`video/test.mp4` is Hailo's official classifier demo asset `example.mp4`
(1280x720, 11.2 s) from
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

### Hardware verification - pending

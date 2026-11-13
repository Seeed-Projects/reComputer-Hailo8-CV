# Test Report - CAS-ViT-S on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | cas_vit_s |
| Task | ImageNet-1k classification |
| Backbone | CAS-ViT-S |
| Parameters | 5.5M |
| Operations | 5.4G |
| Top-1 accuracy (float / Hailo-8) | 79.9% / 79.7% |
| Model Zoo version | v2.20.0 |
| Compile target | hailo8 |

## HEF artifact

| Field | Value |
|---|---|
| File | model/cas_vit_s.hef |
| Size | 12,662,585 bytes |
| SHA256 | `45a467b5347cbdaf1a5dec43be8f338e59664b4411879f26050316d0c15d7121` |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/cas_vit_s.hef |

## Input / output

| Field | Value |
|---|---|
| Input | 384x384x3 BGR uint8 (plain resize; bgr_to_rgb and ImageNet normalization compiled in the HEF) |
| Output | 1000 ImageNet-1k scores (no softmax/argmax in the ALLS recipe) |
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
| Offline harness: BGR passthrough + synthetic 1000-way logits -> Top-K overlay | Pass |
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
# Test Report - LightFace Slim on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | lightface_slim |
| Task | Face detection (single "face" class, WIDER FACE) |
| Backbone | Ultra-Light-Fast-Generic-Face-Detector-1MB (MIT) |
| Parameters | 0.26M |
| Operations | 0.16G |
| Anchors | 4,420 (steps 8/16/32/64, min_sizes [10,16,24]/[32,48]/[64,96]/[128,192,256]) |

## HEF artifact

| Field | Value |
|---|---|
| File | model/lightface_slim.hef |
| Size | 1,907,332 bytes |
| SHA256 | `e448c0f692f9e907bb7328fc2305ced4d3569f1e0aa26d44439d1a6f003e9314` |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/lightface_slim.hef |

## Input / output

| Field | Value |
|---|---|
| Input | 240x320x3 uint8 BGR (aspect-preserving resize, bottom/right pad 0) |
| Output | 8 raw heads: 4 scales x {bbox 4*n, conf 2*n}; no on-chip NMS |
| Decode | SSD (variances 10/5) + softmax over the 2 confidence channels + greedy NMS (0.3) |
| Preview cut | 0.5 confidence (zoo eval uses 0.1) |

## Verification status

| Check | Status |
|---|---|
| Python syntax (py_compile) | Pass |
| Offline decode harness (planted face -> exact box, softmax + NMS paths) | Pass |
| First-inference layout log (vstream names/shapes/ranges) | Pass (logged once, no layout guessing) |
| CI matrix entry added | Pass |

### Hardware verification - pending (assigned to the maintainer)

| Check | Status |
|---|---|
| HEF loads, 8 output vstreams match the documented shapes | Pending |
| Demo video: face boxes align | Pending |
| Single-image REST API | Pending |
| Offline video analysis | Pending |
| USB camera | Pending |
| Official GHCR image re-pull + run | Pending |

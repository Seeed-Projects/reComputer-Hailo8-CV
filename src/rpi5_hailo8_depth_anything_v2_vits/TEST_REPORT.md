# Test Report — Depth-Anything-V2-ViTS on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | depth_anything_v2_vits |
| Task | Zero-shot monocular depth estimation |
| Backbone | ViT-S (Depth-Anything-V2-Small-hf) |
| Parameters | 24.2M |
| Operations | 16.7G |
| Framework source | pytorch (HuggingFace Depth-Anything-V2) |
| Model Zoo version | v2.19.0 |
| Compile target | hailo8 |
| Family | depth_anything (variants: v2_vits / vits planned) |

## HEF artifact

| Field | Value |
|---|---|
| File | model/depth_anything_v2_vits.hef |
| Size | ~35 MB (36,377,888 bytes) |
| SHA256 | 366a3a540a62d9b6e9089850250e29fd9cbd7d814139f881ba63d6bb2a6f8b48 |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/depth_anything_v2_vits.hef |

## Input / output

| Field | Value |
|---|---|
| Input | 224x224x3 RGB, uint8 (normalize_in_net ImageNet RGB) |
| Output | Depth map 224x224x1 (relative, zero-shot) |
| Output dtype | FLOAT32 |
| AbsRel | 0.147 |
| Postprocessing | min-max normalize → INFERNO colormap → alpha blend |

## Verification status

| Check | Status |
|---|---|
| Python syntax (py_compile) | Pass |
| Module structure matches SOP | Pass |
| Dockerfile present + CMD references real HEF | Pass |
| CI matrix entry added | Pass |
| Post-processing (depth colormap + alpha blend) | Pass |

### Hardware verification — pending

| Check | Status |
|---|---|
| HEF loads | Pending |
| Output shape confirmed (224x224x1) | Pending |
| Depth map direction correct (near=dark, far=bright) | Pending |
| Demo video: depth overlay aligns with scene | Pending |
| Official GHCR image re-pull + run | Pending |

## Known points to verify on hardware

- The depth output is **relative** (not metric). Values are per-frame
  min-max normalized for visualization — absolute depth is not available.
- Depth direction: higher output value = farther (verify on hardware; if
  inverted, swap min/max in the colormap).
- The output shape may be (1, 1, 224, 224) [NCHW] or (1, 224, 224, 1) [NHWC]
  — the parser handles both.

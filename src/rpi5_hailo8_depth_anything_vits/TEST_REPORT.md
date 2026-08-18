# Test Report — Depth-Anything-ViTS on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | depth_anything_vits |
| Task | Zero-shot monocular depth estimation |
| Backbone | ViT-S (Depth-Anything V1) |
| Parameters | 24.2M |
| Operations | 16.7G |
| AbsRel | 0.13 |
| Family | depth_anything (variants: v2_vits / vits) |
| Sibling | depth_anything_v2_vits (same I/O, AbsRel 0.147) |

## HEF artifact

| Field | Value |
|---|---|
| File | model/depth_anything_vits.hef |
| Size | ~32 MB |
| SHA256 | fad55db39523f2e89e518bc86d038d1c2a6083e2f36055f39a171ce608ee6bac |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/depth_anything_vits.hef |

## Input / output (same as v2_vits)

| Field | Value |
|---|---|
| Input | 224x224x3 RGB, uint8 |
| Output | Depth map 224x224x1 (relative, zero-shot) |
| Postprocessing | min-max normalize → INFERNO colormap → alpha blend |

## Verification status

| Check | Status |
|---|---|
| Python syntax (py_compile) | Pass |
| Post-processing identical to v2_vits | Pass |
| CI matrix entry added | Pass |

### Hardware verification — pending

| Check | Status |
|---|---|
| HEF loads | Pending |
| Output shape confirmed (224x224x1) | Pending |
| Depth overlay aligns with scene | Pending |
| Official GHCR image re-pull + run | Pending |

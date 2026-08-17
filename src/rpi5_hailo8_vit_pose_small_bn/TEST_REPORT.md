# Test Report — ViTPose-Small-BN on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | vit_pose_small_bn |
| Task | Single-person 2D pose estimation (17 COCO keypoints) |
| Backbone | ViT-Small (ViTPose, BatchNorm variant) |
| Parameters | 24.32M |
| Operations | 17.17G |
| Framework source | pytorch (ViTAE-Transformer/ViTPose) |
| Model Zoo version | v2.19.0 |
| Family | vit_pose (variants: vit_pose_small / vit_pose_small_bn) |
| Sibling | vit_pose_small (same I/O, AP 74.16 vs 72.01) |

## HEF artifact

| Field | Value |
|---|---|
| File | model/vit_pose_small_bn.hef |
| Size | ~31 MB (32,100,744 bytes) |
| SHA256 | 1ccfac57169c7c3a77a3077c75174ba1c3b6a90196b28c54efa25c6713dbe698 |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/vit_pose_small_bn.hef |

## Input / output

| Field | Value |
|---|---|
| Input | 256x192x3 RGB, uint8 |
| Output | Heatmap 64x48x17 (17 keypoints) |
| Postprocessing | argmax per channel → keypoint coords → scale → un-letterbox |

## Verification status

| Check | Status |
|---|---|
| Python syntax (py_compile) | Pass |
| Module structure matches SOP | Pass |
| Dockerfile present + CMD references real HEF | Pass |
| CI matrix entry added | Pass |
| Post-processing identical to vit_pose_small | Pass |

### Hardware verification — pending

| Check | Status |
|---|---|
| HEF loads | Pending |
| Output shape confirmed (64x48x17) | Pending |
| Keypoint positions align with body | Pending |
| Official GHCR image re-pull + run | Pending |

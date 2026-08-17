# Test Report — ViTPose-Small on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | vit_pose_small |
| Task | Single-person 2D pose estimation (17 COCO keypoints) |
| Backbone | ViT-Small (ViTPose) |
| Parameters | 24.29M |
| Operations | 17.17G |
| Framework source | pytorch (ViTAE-Transformer/ViTPose) |
| Model Zoo version | v2.19.0 |
| Compile target | hailo8 |
| Family | vit_pose (variants: vit_pose_small / vit_pose_small_bn planned) |

## HEF artifact

| Field | Value |
|---|---|
| File | model/vit_pose_small.hef |
| Size | ~31 MB (31,721,622 bytes) |
| SHA256 | 7c538a9f1a409298331e6dac905269a8bdbaee405f9c739bda78033f6cbfb450 |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/vit_pose_small.hef |

## Input / output

| Field | Value |
|---|---|
| Input | 256x192x3 RGB, uint8 (normalize_in_net ImageNet RGB mean/std) |
| Output | Heatmap 64x48x17 (17 keypoints) |
| On-chip NMS | N/A (pose, no NMS) |
| Postprocessing | argmax per channel → keypoint coords → scale → un-letterbox |
| num_keypoints | 17 (COCO: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) |

## Verification status

| Check | Status |
|---|---|
| Python syntax (py_compile) | Pass |
| Module structure matches SOP | Pass |
| Dockerfile present + CMD references real HEF | Pass |
| CI matrix entry added | Pass |
| Post-processing ported from official vit_pose_postprocessing.py | Pass |

### Hardware verification — pending

| Check | Status |
|---|---|
| HEF loads | Pending |
| Output shape confirmed (64x48x17) | Pending |
| Keypoint positions align with body | Pending |
| Demo video: skeleton correct | Pending |
| Official GHCR image re-pull + run | Pending |

## Known points to verify on hardware

- The heatmap output shape may be (1, 17, 64, 48) [NCHW] or (1, 64, 48, 17)
  [NHWC] — the parser handles both.
- Single-person assumption: the person must be centered in the crop for good
  results. The bundled test video may need to show a single person.

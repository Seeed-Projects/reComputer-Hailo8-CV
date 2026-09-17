# Test Report — MSPN RegNetX-800MF on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | mspn_regnetx_800mf |
| Task | Single-person 2D pose estimation (17 COCO keypoints) |
| Backbone | RegNetX-800MF (MSPN, Multi-Stage Pose Network) |
| Parameters | 7.17M |
| Operations | 2.94G |
| Framework source | pytorch (open-mmlab/mmpose) |
| Model Zoo version | v2.19.0 |
| Compile target | hailo8 |
| Family | mspn (variants: mspn_regnetx_800mf) |

## HEF artifact

| Field | Value |
|---|---|
| File | model/mspn_regnetx_800mf.hef |
| Size | 6,612,716 bytes |
| SHA256 | ba9028310008f0d7601497230e2adf6f2816e78d532b71f3d489237b154d1534 |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/mspn_regnetx_800mf.hef |

## Input / output

| Field | Value |
|---|---|
| Input | 256x192x3 RGB, uint8 (normalize_in_net ImageNet RGB mean/std) |
| Output | Heatmap 64x48x17 (17 keypoints); parsed from NHWC or NCHW |
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
| Post-processing verified against Model Zoo `mspn_postprocessing.py` (argmax decode, layout transpose) | Pass |

### Hardware verification — pending

| Check | Status |
|---|---|
| HEF loads on Hailo-8 | Pending |
| Output shape confirmed (64x48x17) | Pending |
| Keypoint positions align with body | Pending |
| Demo video: skeleton correct | Pending |
| Official GHCR image re-pull + run | Pending |

## Known points to verify on hardware

- The raw vstream layout may be (1, 64, 48, 17) [NHWC] or (1, 17, 64, 48)
  [NCHW] — the parser handles both and logs the actual shape once at the
  first inference.
- Keypoint score range after HailoRT dequantization must be confirmed on
  hardware; the default confidence threshold is 0.30.
- The Model Zoo reference applies a 5×5 Gaussian blur and a quarter-pixel
  shift before/after argmax; this module skips both. If keypoint jitter is
  visible on hardware, port those two steps.
- Single-person assumption: the person must be centered in the crop for good
  results. The bundled test video may need to show a single person.
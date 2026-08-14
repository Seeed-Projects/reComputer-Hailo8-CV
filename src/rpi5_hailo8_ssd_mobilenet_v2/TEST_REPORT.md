# Test Report — SSD MobileNet V2 on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | ssd_mobilenet_v2 |
| Task | Object detection (COCO 80 thing classes) |
| Backbone | MobileNet V2 |
| Parameters | 4.46M |
| Operations | 1.52G |
| Framework source | tensorflow (TF Object Detection API) |
| Model Zoo version | v2.19.0 |
| Compile target | hailo8 |
| Family | ssd (variants: ssd_mobilenet_v1 / ssd_mobilenet_v2) |
| Sibling | ssd_mobilenet_v1 (same I/O contract, V1 backbone, heavier) |

## HEF artifact

| Field | Value |
|---|---|
| File | model/ssd_mobilenet_v2.hef |
| Size | ~5.6 MB (5,814,255 bytes) |
| SHA256 | aad32ba59cd8768f382b6b8e5079b3c3c4c3efde1bfaeb3282d7c4f5e0f0cb2a |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/ssd_mobilenet_v2.hef |

## Input / output (from Model Zoo network YAML + base/ssd.yaml)

| Field | Value |
|---|---|
| Input | 300x300x3 RGB, uint8 (normalize_in_net mean=127.5/std=127.5) |
| Padding | color 0 (black) |
| On-chip | nms=true, meta_arch=ssd (Hailo HPP NMS) |
| Post-NMS output | single vstream, shape 90x8x1 |
| Per-detection row | [ymin, xmin, ymax, xmax, score, ...], normalized [0,1] to 300x300 input |
| Output dtype | FLOAT32 (ragged object array in HailoRT NMS-by-score form) |
| num_classes | 90 (COCO category IDs 1..90 via labels_offset=1; 10 unused IDs) |
| score_threshold (eval) | 0.3 |
| nms_iou_thresh (eval) | 0.6 |

## Verification status

| Check | Status |
|---|---|
| Python syntax (py_compile) | Pass |
| Module structure matches SOP | Pass |
| Dockerfile present + CMD references real HEF | Pass |
| CI matrix entry added | Pass |
| HEF input/output taken from Model Zoo source (not guessed) | Pass |
| First-inference vstream logging implemented (SOP §10) | Pass |
| Post-processing identical to verified V1 build (on-chip NMS HPP, ragged-safe) | Pass |

### Hardware verification — pending

| Check | Status |
|---|---|
| HEF loads, no driver/firmware mismatch | Pending |
| Output vstream type/shape confirmed from first-inference log | Pending |
| Class mapping correct (90-class, labels_offset=1, N/A gaps not drawn) | Pending |
| Demo video: boxes align with objects, correct classes | Pending |
| Official GHCR image re-pull + run | Pending |

## Known points to verify on hardware

- The post-NMS shape `90x8x1` means max 1 detection per class (8 values per det,
  first 5 parsed as [ymin, xmin, ymax, xmax, score]). Verify the count is
  acceptable for your use case.
- HailoRT returns the NMS vstream as a ragged per-class list (NMS-by-score).
- `nms_thresh` is ignored (NMS on-chip); only `obj_thresh` filters the final
  output.

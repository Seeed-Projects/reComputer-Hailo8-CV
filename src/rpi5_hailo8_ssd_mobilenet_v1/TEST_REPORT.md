# Test Report — SSD MobileNet V1 on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | ssd_mobilenet_v1 |
| Task | Object detection (COCO 80 thing classes) |
| Backbone | MobileNet V1 |
| Parameters | 6.79M |
| Operations | 2.5G |
| Framework source | tensorflow (TF Object Detection API) |
| Model Zoo version | v2.19.0 |
| Compile target | hailo8 |
| Family | ssd (variants: ssd_mobilenet_v1 / ssd_mobilenet_v2 planned) |

## HEF artifact

| Field | Value |
|---|---|
| File | model/ssd_mobilenet_v1.hef |
| Size | ~6.4 MB (6,704,724 bytes) |
| SHA256 | 51234b4a5601b377cb955af45f6af14e8aba7c2c22b78baaa5504fa46ccff07a |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/ssd_mobilenet_v1.hef |

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

## Runtime baseline

| Field | Value |
|---|---|
| HailoRT | 4.23.x |
| Python | 3.11, aarch64 |
| Wheel | hailort-4.23.0-cp311-cp311-linux_aarch64.whl |
| Kernel module | hailo_pci |
| Device node | /dev/hailo0 |

## Verification status

Static/code-level verification on the development host (no Hailo-8 hardware
locally). Hardware fields are left for the on-device run.

| Check | Status |
|---|---|
| Python syntax (py_compile) | Pass |
| Module structure matches SOP | Pass |
| Dockerfile present + CMD references real HEF | Pass |
| CI matrix entry added | Pass |
| HEF input/output taken from Model Zoo source (not guessed) | Pass |
| First-inference vstream logging implemented (SOP §10) | Pass |
| Post-processing ported from official tf_postproc_nms (on-chip NMS HPP, ragged-safe) | Pass |

### Hardware verification — pending

To be filled after the SSH pull-and-run on CM5 + Hailo-8:

| Check | Status |
|---|---|
| PCI device / `/dev/hailo0` present | Pending |
| HEF loads, no driver/firmware mismatch | Pending |
| Output vstream type/shape confirmed from first-inference log | Pending |
| Ragged NMS-by-score layout confirmed; parser path correct | Pending |
| Class mapping correct (90-class, labels_offset=1, N/A gaps not drawn) | Pending |
| Demo video: boxes align with objects, correct classes | Pending |
| Single-image REST API: class/confidence/box correct | Pending |
| Official GHCR image re-pull + run | Pending |

## Known points to verify on hardware

- The post-NMS shape `90x8x1` means max 1 detection per class (8 values per det,
  first 5 parsed as [ymin, xmin, ymax, xmax, score]). This is a low max-dets
  setting — the demo may show fewer boxes than other models. Verify the count
  is acceptable for your use case.
- HailoRT returns the NMS vstream as a ragged per-class list (NMS-by-score).
  The parser handles ragged/object/dense layouts and logs the raw type/shape.
- `nms_thresh` is ignored (NMS on-chip); only `obj_thresh` filters the final
  output. Eval uses 0.3 / 0.6.

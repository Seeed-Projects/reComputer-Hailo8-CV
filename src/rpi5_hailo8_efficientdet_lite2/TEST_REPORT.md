# Test Report — EfficientDet-Lite2 on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | efficientdet_lite2 |
| Task | Object detection (COCO 80 thing classes) |
| Backbone | EfficientDet-Lite2 (BiFPN) |
| Parameters | 5.93M |
| Operations | 6.84G |
| Framework source | tensorflow (google/automl) |
| Model Zoo version | v2.19.0 |
| Compile target | hailo8 |
| Family | efficientdet (variants: lite0 / lite1 / lite2) |
| Sibling | efficientdet_lite0 (320x320), efficientdet_lite1 (384x384) — same I/O contract |

## HEF artifact

| Field | Value |
|---|---|
| File | model/efficientdet_lite2.hef |
| Size | ~14 MB (14,313,963 bytes) |
| SHA256 | 937c3e6c6d04e25b22aedc36dd192aff8d769fcec35508fdfcb47390e4fcdaf4 |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/efficientdet_lite2.hef |

## Input / output (from Model Zoo network YAML + base/efficientdet_lite.yaml)

| Field | Value |
|---|---|
| Input | 448x448x3 RGB, uint8 (normalize_in_net mean=127/std=128) |
| Padding | color 127 (gray) |
| On-chip | nms=true, sigmoid=true, hpp=true (Hailo HPP NMS) |
| Post-NMS output | single vstream, shape 89x5x100 = (num_classes, 5, max_dets) |
| Per-detection row | [ymin, xmin, ymax, xmax, score], normalized [0,1] to 448x448 input |
| Output dtype | FLOAT32 (ragged object array in HailoRT NMS-by-score form) |
| num_classes | 89 (COCO category IDs 1..89 via labels_offset=1; 10 unused IDs) |
| score_threshold (eval) | 0.001 |
| nms_iou_thresh (eval) | 0.5 |

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
| Post-processing identical to verified Lite0 build (incl. ragged NMS-by-score fix) | Pass |

### Hardware verification — pending

To be filled after the SSH pull-and-run on CM5 + Hailo-8:

| Check | Status |
|---|---|
| PCI device / `/dev/hailo0` present | Pending |
| `hailortcli fw-control identify` succeeds | Pending |
| HEF loads, no driver/firmware mismatch | Pending |
| Output vstream type/shape confirmed from first-inference log | Pending |
| Ragged NMS-by-score layout confirmed; parser path correct | Pending |
| Class mapping correct (labels not shifted; "N/A" gaps not drawn) | Pending |
| Demo video: boxes align with objects, correct classes | Pending |
| Single-image REST API: class/confidence/box correct | Pending |
| Offline video analysis: upload/progress/download | Pending |
| USB camera: real-time preview, latency, stability | Pending |
| Official GHCR image re-pull + run | Pending |

## Known points to verify on hardware

- HailoRT returns the NMS vstream as a ragged per-class list (NMS-by-score) —
  the same form as Lite0/Lite1. The parser handles ragged/object/dense layouts
  and logs the raw type/shape on first inference; confirm the chosen path
  produces sensible boxes.
- Class mapping assumes COCO category IDs (cls_id+1, labels_offset=1). The 10
  unused COCO IDs are "N/A" and skipped.
- `nms_thresh` is ignored (NMS is on-chip); only `obj_thresh` filters the final
  output.

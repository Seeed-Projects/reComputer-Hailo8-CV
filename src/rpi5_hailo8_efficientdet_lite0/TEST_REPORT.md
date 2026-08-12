# Test Report — EfficientDet-Lite0 on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | efficientdet_lite0 |
| Task | Object detection (COCO 80 thing classes) |
| Backbone | EfficientDet-Lite0 (BiFPN) |
| Parameters | 3.56M |
| Operations | 1.94G |
| Framework source | tensorflow (google/automl) |
| Model Zoo version | v2.19.0 |
| Compile target | hailo8 |
| Family | efficientdet (variants: lite0 / lite1 / lite2 planned) |

## HEF artifact

| Field | Value |
|---|---|
| File | model/efficientdet_lite0.hef |
| Size | ~9.3 MB (9,733,034 bytes) |
| SHA256 | d017f8754bd2a79a0406396a815d9dfb8a34108b8f6c0f09ccbf8b0178a0c994 |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/efficientdet_lite0.hef |

## Input / output (from Model Zoo network YAML + base/efficientdet_lite.yaml)

| Field | Value |
|---|---|
| Input | 320x320x3 RGB, uint8 (normalize_in_net mean=127/std=128) |
| Padding | color 127 (gray) |
| On-chip | nms=true, sigmoid=true, hpp=true (Hailo HPP NMS) |
| Post-NMS output | single vstream, shape 89x5x100 = (num_classes, 5, max_dets) |
| Per-detection row | [ymin, xmin, ymax, xmax, score], normalized [0,1] to 320x320 input |
| Output dtype | FLOAT32 |
| num_classes | 89 (COCO category IDs 1..89 via labels_offset=1; 10 unused IDs) |
| score_threshold (eval) | 0.001 |
| nms_iou_thresh (eval) | 0.5 |
| meta_arch | efficientdet (preprocessing meta_arch: mobilenet_ssd_ar) |

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
| Post-processing ported from official tf_postproc_nms (on-chip NMS HPP) | Pass |

### Hardware verification — pending

To be filled after the SSH pull-and-run on CM5 + Hailo-8:

| Check | Status |
|---|---|
| PCI device / `/dev/hailo0` present | Pending |
| `hailortcli fw-control identify` succeeds | Pending |
| HEF loads, no driver/firmware mismatch | Pending |
| Output vstream type/shape confirmed from first-inference log | Pending |
| HPP NMS layout (list/object/dense) confirmed; parser path correct | Pending |
| Class mapping correct (labels not shifted; "N/A" gaps not drawn) | Pending |
| Demo video: boxes align with objects, correct classes | Pending |
| Single-image REST API: class/confidence/box correct | Pending |
| Offline video analysis: upload/progress/download | Pending |
| USB camera: real-time preview, latency, stability | Pending |
| Official GHCR image re-pull + run | Pending |

## Known points to verify on hardware

- The NMS vstream may arrive as a per-class list (NMS-by-score), an object
  array, or dense float32 `(1,89,5,100)` / `(1,89,100,5)`. The parser handles
  all three and logs the raw type/shape on first inference; confirm the chosen
  path produces sensible boxes.
- Class mapping assumes COCO category IDs (cls_id+1, labels_offset=1). The 10
  unused COCO IDs are mapped to "N/A" and skipped. If labels look systematically
  shifted, the 89-class layout may differ — re-derive from the first-inference
  log.
- `nms_thresh` is ignored (NMS is on-chip); only `obj_thresh` filters the final
  output. The eval uses 0.001 — the demo defaults to 0.25 for a cleaner
  preview (lower the slider to inspect everything).

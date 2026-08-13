# Test Report — NanoDet-RepVGG on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | nanodet_repvgg |
| Task | Object detection (COCO 80 classes) |
| Backbone | NanoDet + RepVGG |
| Parameters | 6.74M |
| Operations | 11.28G |
| Framework source | pytorch (RangiLyu/nanodet) |
| Model Zoo version | v2.19.0 |
| Compile target | hailo8 |
| Family | nanodet (variants: nanodet_repvgg / nanodet_repvgg_a12 / nanodet_repvgg_a1_640 planned) |

## HEF artifact

| Field | Value |
|---|---|
| File | model/nanodet_repvgg.hef |
| Size | ~10.7 MB (10,721,749 bytes) |
| SHA256 | 9eb2fd97397356297a114790eafadeda48a7ad072d4f51b436b9adc139dd1baa |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/nanodet_repvgg.hef |

## Input / output (from Model Zoo network YAML + base/nanodet.yaml)

| Field | Value |
|---|---|
| Input | 416x416x3 BGR, uint8 (input_conversion bgr_to_rgb + normalize_in_net ImageNet RGB mean/std) |
| Padding | color 0 (black) |
| On-chip | nms=true, hpp=true, meta_arch=yolov8 (Hailo HPP NMS) |
| Post-NMS output | single vstream, shape 80x5x100 = (num_classes, 5, max_dets) |
| Per-detection row | [ymin, xmin, ymax, xmax, score], normalized [0,1] to 416x416 input |
| Output dtype | FLOAT32 (ragged object array in HailoRT NMS-by-score form) |
| num_classes | 80 (COCO, 0-indexed via meta_arch=yolov8 — no labels_offset) |
| score_threshold (eval) | 0.05 |
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
| `hailortcli fw-control identify` succeeds | Pending |
| HEF loads, no driver/firmware mismatch | Pending |
| Output vstream type/shape confirmed from first-inference log | Pending |
| Ragged NMS-by-score layout confirmed; parser path correct | Pending |
| Class mapping correct (80-class, 0-indexed; no shift) | Pending |
| BGR feed correct (boxes/colours not degraded; if bad, swap to RGB) | Pending |
| Demo video: boxes align with objects, correct classes | Pending |
| Single-image REST API: class/confidence/box correct | Pending |
| Offline video analysis: upload/progress/download | Pending |
| USB camera: real-time preview, latency, stability | Pending |
| Official GHCR image re-pull + run | Pending |

## Known points to verify on hardware

- HailoRT returns the NMS vstream as a ragged per-class list (NMS-by-score). The
  parser handles ragged/object/dense layouts and logs the raw type/shape on first
  inference; confirm the chosen path produces sensible boxes.
- Input is **BGR** (the `.alls` has `input_conversion(bgr_to_rgb)`, so the HEF
  expects BGR and converts to RGB internally). If detections look systematically
  off (wrong colours / poor), try feeding RGB (add cvtColor) and re-check.
- `nms_thresh` is ignored (NMS is on-chip); only `obj_thresh` filters the final
  output. Eval uses 0.05 / 0.6 — demo defaults to 0.25 / 0.45 for a cleaner
  preview.

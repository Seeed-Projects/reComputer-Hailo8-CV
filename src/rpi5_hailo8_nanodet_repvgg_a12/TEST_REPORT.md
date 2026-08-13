# Test Report — NanoDet-RepVGG-a12 on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | nanodet_repvgg_a12 |
| Task | Object detection (COCO 80 classes) |
| Architecture | YOLOX-based (base/yolox.yaml) — despite the "nanodet" name |
| Backbone | RepVGG-A12 |
| Parameters | 5.13M |
| Operations | 28.23G |
| Framework source | pytorch (Megvii-BaseDetection/YOLOX) |
| Model Zoo version | v2.19.0 |
| Compile target | hailo8 |
| Family | nanodet (variants: nanodet_repvgg / nanodet_repvgg_a12 / nanodet_repvgg_a1_640) |

## HEF artifact

| Field | Value |
|---|---|
| File | model/nanodet_repvgg_a12.hef |
| Size | ~7.5 MB (7,801,762 bytes) |
| SHA256 | c6d99b9df673b1c60ef109ad2c5f18cbf809cceabdbbfa21fd5f0b9376de0b59 |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/nanodet_repvgg_a12.hef |

## Input / output (from Model Zoo network YAML + base/yolox.yaml)

| Field | Value |
|---|---|
| Input | 640x640x3 BGR, uint8 (input_conversion bgr_to_rgb + no-op normalize_in_net mean=0/std=1) |
| Padding | color 114 (gray, YOLOX convention) |
| On-chip | nms=true, hpp=true, meta_arch=yolox (Hailo HPP NMS) |
| Post-NMS output | single vstream, shape 80x5x100 = (num_classes, 5, max_dets) |
| Per-detection row | [ymin, xmin, ymax, xmax, score], normalized [0,1] to 640x640 input |
| Output dtype | FLOAT32 (ragged object array in HailoRT NMS-by-score form) |
| num_classes | 80 (COCO, 0-indexed; labels_offset=1 eval-only) |
| score_threshold (eval) | 0.01 |
| nms_iou_thresh (eval) | 0.65 |

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
| Post-processing identical to nanodet_repvgg (on-chip NMS HPP, ragged-safe) | Pass |

### Hardware verification — pending

To be filled after the SSH pull-and-run on CM5 + Hailo-8:

| Check | Status |
|---|---|
| PCI device / `/dev/hailo0` present | Pending |
| `hailortcli fw-control identify` succeeds | Pending |
| HEF loads, no driver/firmware mismatch | Pending |
| Output vstream type/shape confirmed from first-inference log | Pending |
| Ragged NMS-by-score layout confirmed; parser path correct | Pending |
| Class mapping correct (80-class 0-indexed; labels_offset eval-only) | Pending |
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
- Input is **BGR** (`.alls` `input_conversion(bgr_to_rgb)`). If detections look
  systematically off, try feeding RGB (add cvtColor) and re-check.
- Class mapping assumes 0-indexed 80-class (labels_offset=1 is eval-only). If
  labels are shifted, re-derive from the first-inference log.
- `nms_thresh` is ignored (NMS on-chip); only `obj_thresh` filters the final
  output.

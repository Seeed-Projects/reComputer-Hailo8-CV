# Test Report — NanoDet-RepVGG-a1-640 on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | nanodet_repvgg_a1_640 |
| Task | Object detection (COCO 80 classes) |
| Architecture | NanoDet + RepVGG-A1 (base/nanodet.yaml) |
| Parameters | 10.79M |
| Operations | 42.8G |
| Framework source | pytorch (RangiLyu/nanodet) |
| Model Zoo version | v2.19.0 |
| Compile target | hailo8 |
| Family | nanodet (variants: nanodet_repvgg / nanodet_repvgg_a12 / nanodet_repvgg_a1_640) |

## HEF artifact

| Field | Value |
|---|---|
| File | model/nanodet_repvgg_a1_640.hef |
| Size | ~9.8 MB (10,280,822 bytes) |
| SHA256 | 172a4e9ffb2ab34b3d4202620018cc4c6dd05909d00e7418f656c589d360600f |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/nanodet_repvgg_a1_640.hef |

## Input / output (from Model Zoo network YAML + base/nanodet.yaml)

| Field | Value |
|---|---|
| Input | 640x640x3 BGR, uint8 (input_conversion bgr_to_rgb + normalize_in_net) |
| Padding | color 0 (black) |
| On-chip | nms=true, hpp=true, meta_arch=yolov8 (Hailo HPP NMS) |
| Post-NMS output | single vstream, shape 80x5x100 = (num_classes, 5, max_dets) |
| Per-detection row | [ymin, xmin, ymax, xmax, score], normalized [0,1] to 640x640 input |
| Output dtype | FLOAT32 (ragged object array in HailoRT NMS-by-score form) |
| num_classes | 80 (COCO, 0-indexed via meta_arch=yolov8) |
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
| Class mapping correct (80-class, 0-indexed) | Pending |
| BGR feed correct (boxes/colours not degraded; if bad, swap to RGB) | Pending |
| Demo video: boxes align with objects, correct classes | Pending |
| Single-image REST API: class/confidence/box correct | Pending |
| Official GHCR image re-pull + run | Pending |

## Known points to verify on hardware

- HailoRT returns the NMS vstream as a ragged per-class list (NMS-by-score). The
  parser handles ragged/object/dense layouts and logs the raw type/shape on first
  inference.
- Input is **BGR** (`.alls` `input_conversion(bgr_to_rgb)`). If detections look
  systematically off, try feeding RGB (add cvtColor) and re-check.
- `nms_thresh` is ignored (NMS on-chip); only `obj_thresh` filters the final
  output.

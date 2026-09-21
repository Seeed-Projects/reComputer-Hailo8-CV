# Test Report — YOLOv5n-seg on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | yolov5n_seg |
| Task | Instance segmentation (COCO 80 classes) |
| Architecture | YOLOv5n-seg (anchor-based, 3 anchors x 3 strides) |
| Parameters | 1.99M |
| Model Zoo version | v2.19.0 |
| Compile target | hailo8 |

## HEF artifact

| Field | Value |
|---|---|
| File | model/yolov5n_seg.hef |
| Size | 4,723,980 bytes |
| SHA256 | 2095dac5a3f1d7de2ecfc7fb1950948ebcdcbf96246425a3239f81a604f519d4 |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/yolov5n_seg.hef |

## Input / output

| Field | Value |
|---|---|
| Input | 640x640x3 RGB (normalize_in_net mean 0 / std 255) |
| Letterbox padding | 114 (gray, YOLO convention) |
| Output | proto (160x160x32) + 3 detection heads (351 channels, strides 32/16/8) |
| On-chip NMS | No |
| Postprocessing | anchor decode + per-class NMS + sigmoid(coeffs @ proto) masks |

## Verification status

| Check | Status |
|---|---|
| Python syntax (py_compile) | Pass |
| Module structure matches SOP | Pass |
| Dockerfile present + CMD references real HEF | Pass |
| CI matrix entry added | Pass |
| Post-processing verified with synthetic tensors (anchor decode, NMS, mask assembly) | Pass |

### Hardware verification — pending

| Check | Status |
|---|---|
| HEF loads on Hailo-8 | Pending |
| Output vstream names/shapes confirmed on hardware | Pending |
| Box and mask alignment on the demo video | Pending |
| Official GHCR image re-pull + run | Pending |

## Known points to verify on hardware

- First inference prints every vstream name/shape; confirm the 351-channel
  heads map to strides 32/16/8 in that order and that the proto head is
  160x160x32.
- Anchor sizes/order are the Model Zoo `base/yolo.yaml` values
  (strides 8/16/32; sizes per stride as listed there); confirm decoded box
  positions against a known image.
- Mask coefficients scale with objectness in the Model Zoo NMS
  implementation, so mask intensity depends on the objectness logit range
  returned by the HEF after dequantization.

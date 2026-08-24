# YOLACT RegNetX-800MF on Raspberry Pi 5 / CM5 + Hailo-8

This module runs YOLACT-RegNetX-800MF instance segmentation (COCO 80 classes,
UC Davis) with **CPU-decoded instance segmentation** (`meta_arch=yolact`).
The HEF exposes 16 raw heads — prototype masks, box regression, mask
coefficients and confidence — and the app performs the full decode chain:
anchor generation, SSD-style box decode, Fast NMS, and mask assembly
(`proto @ coeffs` + sigmoid + crop), ported 1:1 from the Hailo Model Zoo.

The FastAPI service supports images, video files, USB cameras, an MJPEG
preview with mask overlays, and REST prediction endpoints.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-8 PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 4.23.x |
| Python | 3.11, aarch64 |
| Input | 512x512x3 RGB (normalize_in_net ImageNet RGB mean/std) |
| Output | 16 heads: proto (128x128x32) + 5 scales x {bbox 36, mask 288, conf 729} |
| Priors | 49,104 anchors (9/cell over 64/32/16/8/4 feature maps) |
| Classes | 80 (COCO, Model Zoo class-index order) |
| Parameters | 28.3M |
| Operations | 116.75G |
| HEF | Model Zoo v2.19.0, Hailo-8 |

## Build

```bash
sudo docker build -f docker/hailo8/yolact_regnetx_800mf.dockerfile \
  -t yolact_regnetx_800mf:latest \
  src/rpi5_hailo8_yolact_regnetx_800mf
```

## Run

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  yolact_regnetx_800mf:latest \
  python web_detection.py --model_path model/yolact_regnetx_800mf.hef --video_path video/test.mp4
```

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/yolact_regnetx_800mf/predict" \
  -F "file=@test.jpg"
```

| Endpoint | Method | Description |
|---|---|---|
| `/api/models/yolact_regnetx_800mf/predict` | POST | Detections + per-object mask area (JSON) |
| `/api/video_feed` | GET | MJPEG preview stream (masks + boxes) |

## Implementation notes

- CPU post-processing (`meta_arch=yolact`): no on-chip NMS. The Detect
  pipeline (49,104 anchors → decode → per-instance argmax class → Fast NMS,
  top-200/class, max 100 detections) is a numpy port of
  `instance_segmentation_postprocessing.py` from the Model Zoo.
- `normalize_in_net` ImageNet RGB ([123.68, 116.78, 103.94] /
  [58.4, 57.12, 57.38]); feed raw uint8 RGB — no manual normalization.
- Preprocessing is plain bilinear resize to 512x512 (no letterbox, no pad),
  matching the Model Zoo's `mobilenet_ssd` preprocessor for this network.
- Masks: `sigmoid(proto @ coeffs^T)` crop to each box; visualized with
  per-class colors, `mask_thresh=0.5` (YAML `mask_threshold`).
- Class mapping: class index (0..79) → `CLASS_NAMES_COCO` directly.
- The first-inference log prints every output vstream shape so the 16-head
  layout can be verified against the network YAML on hardware.

## Model source

[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_instance_segmentation.rst)
(model `yolact_regnetx_800mf`, source: [dbolya/yolact](https://github.com/dbolya/yolact)).
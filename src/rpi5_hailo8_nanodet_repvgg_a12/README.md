# NanoDet-RepVGG-a12 on Raspberry Pi 5 / CM5 + Hailo-8

This module runs nanodet_repvgg_a12 object detection (COCO 80 classes, YOLOX-based)
with **on-chip NMS** (Hailo HPP, `meta_arch=yolox`). Despite the "nanodet" name
this variant is built on the YOLOX architecture (base/yolox.yaml), 640x640 input.
The HEF performs NMS on-device and emits already-decoded detections; the app only
parses the post-NMS tensor. The FastAPI service supports images, video files, USB
cameras, an MJPEG preview, and REST prediction endpoints.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-8 PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 4.23.x |
| Python | 3.11, aarch64 |
| Input | 640x640x3 **BGR** (input_conversion bgr_to_rgb + no-op normalize_in_net) |
| Output | on-chip NMS tensor, post-NMS shape 80x5x100 |
| Classes | 80 (COCO, 0-indexed; labels_offset=1 is eval-only) |
| Parameters | 5.13M |
| Operations | 28.23G |
| HEF | Model Zoo v2.19.0, Hailo-8 |

The host driver, firmware, `libhailort.so`, and Python wheel must use the same
HailoRT major/minor version.

## Build

From the repository root:

```bash
sudo docker build -f docker/hailo8/nanodet_repvgg_a12.dockerfile \
  -t nanodet_repvgg_a12:latest \
  src/rpi5_hailo8_nanodet_repvgg_a12
```

## Run the demo video

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  nanodet_repvgg_a12:latest \
  python web_detection.py --model_path model/nanodet_repvgg_a12.hef --video_path video/test.mp4
```

Open `http://<PI_IP>:8000`. For a USB camera, mount `/dev/video0` and replace
`--video_path video/test.mp4` with `--camera_id 0`.

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/nanodet_repvgg_a12/predict" \
  -F "file=@test.jpg"
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/api/models/nanodet_repvgg_a12/predict` | POST | Detections (JSON) |
| `/api/video_feed` | GET | MJPEG preview stream |

## Implementation notes

- Same on-chip NMS (HPP) post-processing as nanodet_repvgg (80-class HPP parse),
  but this variant uses base/yolox.yaml: `meta_arch=yolox`, 640x640 input,
  `padding_color=114` (gray, YOLOX convention).
- The HEF runs NMS on-chip (`device_pre_post_layers: nms=true, hpp=true`); the
  app only parses the post-NMS tensor (per the official `tf_postproc_nms`), so
  `nms_thresh` is ignored (kept for API parity).
- Post-NMS rows are `[ymin, xmin, ymax, xmax, score]`, normalized to [0,1] of
  the 640x640 letterboxed input; the app scales to pixels and un-letterboxes.
- `normalize_in_net` no-op (mean=0/std=1) + on-chip `input_conversion(bgr_to_rgb)`:
  the app letterboxes with gray (114) padding and feeds **raw uint8 BGR pixels**
  — no manual normalization, no cvtColor.
- HailoRT returns the NMS vstream as a ragged per-class list (NMS-by-score); the
  parser handles that plus object/dense layouts. First inference logs the raw
  type/shape for on-device verification (SOP §10).
- Class mapping: `cls_id` (0..79) → standard COCO 80-class list directly
  (labels_offset=1 is eval-only; if labels look shifted on hardware, re-derive).

## Model source

The Hailo-8 HEF comes from the
[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst)
(model `nanodet_repvgg_a12`).

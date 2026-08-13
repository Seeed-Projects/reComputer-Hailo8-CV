# NanoDet-RepVGG on Raspberry Pi 5 / CM5 + Hailo-8

This module runs NanoDet-RepVGG object detection (COCO 80 classes) with
**on-chip NMS** (Hailo HPP, `meta_arch=yolov8`). The HEF performs NMS on-device
and emits already-decoded detections; the app only parses the post-NMS tensor.
The FastAPI service supports images, video files, USB cameras, an MJPEG preview,
and REST prediction endpoints.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-8 PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 4.23.x |
| Python | 3.11, aarch64 |
| Input | 416x416x3 **BGR** (input_conversion bgr_to_rgb + normalize_in_net compiled into the HEF) |
| Output | on-chip NMS tensor, post-NMS shape 80x5x100 |
| Classes | 80 (COCO, 0-indexed via meta_arch=yolov8 — no labels_offset) |
| Parameters | 6.74M |
| Operations | 11.28G |
| HEF | Model Zoo v2.19.0, Hailo-8 |

The host driver, firmware, `libhailort.so`, and Python wheel must use the same
HailoRT major/minor version.

## Build

From the repository root:

```bash
sudo docker build -f docker/hailo8/nanodet_repvgg.dockerfile \
  -t nanodet_repvgg:latest \
  src/rpi5_hailo8_nanodet_repvgg
```

## Run the demo video

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  nanodet_repvgg:latest \
  python web_detection.py --model_path model/nanodet_repvgg.hef --video_path video/test.mp4
```

Open `http://<PI_IP>:8000`. For a USB camera, mount `/dev/video0` and replace
`--video_path video/test.mp4` with `--camera_id 0`.

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/nanodet_repvgg/predict" \
  -F "file=@test.jpg"
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/api/models/nanodet_repvgg/predict` | POST | Detections (JSON) |
| `/api/video_feed` | GET | MJPEG preview stream |

## Implementation notes

- The HEF runs NMS on-chip (`device_pre_post_layers: nms=true, hpp=true`,
  `meta_arch=yolov8`); the app only parses the post-NMS tensor (per the official
  `tf_postproc_nms`), so `nms_thresh` is ignored (kept for API parity).
- Post-NMS rows are `[ymin, xmin, ymax, xmax, score]`, normalized to [0,1] of
  the 416x416 letterboxed input; the app scales to pixels and un-letterboxes.
- `normalize_in_net` (ImageNet RGB mean/std) + on-chip `input_conversion(bgr_to_rgb)`:
  the app letterboxes with black (0) padding and feeds **raw uint8 BGR pixels** —
  no manual normalization, no cvtColor (the HEF converts BGR→RGB internally).
- HailoRT returns the NMS vstream as a ragged per-class list (NMS-by-score); the
  parser handles that plus object/dense layouts. First inference logs the raw
  type/shape for on-device verification (SOP §10).
- Class mapping: `cls_id` (0..79) → standard COCO 80-class list directly
  (`meta_arch=yolov8`, 0-indexed, no labels_offset, no gaps).

## Model source

The Hailo-8 HEF comes from the
[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst)
(model `nanodet_repvgg`).

# EfficientDet-Lite2 on Raspberry Pi 5 / CM5 + Hailo-8

This module runs EfficientDet-Lite2 object detection (COCO 80 thing classes)
with **on-chip NMS** (Hailo HPP). Same architecture as the Lite0/Lite1 builds
but with the largest backbone and 448x448 input → highest accuracy of the
three. The HEF performs NMS + sigmoid on-device and emits already-decoded
detections; the app only parses the post-NMS tensor. The FastAPI service
supports images, video files, USB cameras, an MJPEG preview, and REST
prediction endpoints.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-8 PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 4.23.x |
| Python | 3.11, aarch64 |
| Input | 448x448x3 RGB (normalization compiled into the HEF: mean=127, std=128) |
| Output | on-chip NMS tensor, post-NMS shape 89x5x100 |
| Classes | 89 slots (COCO category IDs 1..89 via labels_offset=1; 10 unused IDs) |
| Parameters | 5.93M |
| Operations | 6.84G |
| HEF | Model Zoo v2.19.0, Hailo-8 |

The host driver, firmware, `libhailort.so`, and Python wheel must use the same
HailoRT major/minor version.

## Build

From the repository root:

```bash
sudo docker build -f docker/hailo8/efficientdet_lite2.dockerfile \
  -t efficientdet_lite2:latest \
  src/rpi5_hailo8_efficientdet_lite2
```

## Run the demo video

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  efficientdet_lite2:latest \
  python web_detection.py --model_path model/efficientdet_lite2.hef --video_path video/test.mp4
```

Open `http://<PI_IP>:8000`. For a USB camera, mount `/dev/video0` and replace
`--video_path video/test.mp4` with `--camera_id 0`.

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/efficientdet_lite2/predict" \
  -F "file=@test.jpg"
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/api/models/efficientdet_lite2/predict` | POST | Detections (JSON) |
| `/api/video_feed` | GET | MJPEG preview stream |

## Implementation notes

- Identical input/output contract and post-processing to the Lite0/Lite1
  builds; only the backbone and input size differ (448x448, 5.93M params →
  highest accuracy of the three).
- The HEF runs NMS on-chip (`device_pre_post_layers: nms=true, sigmoid=true,
  hpp=true`); the app only parses the post-NMS tensor (per the official
  `tf_postproc_nms`), so `nms_thresh` is ignored (kept for API parity).
- Post-NMS rows are `[ymin, xmin, ymax, xmax, score]`, normalized to [0,1] of
  the 448x448 letterboxed input; the app scales to pixels and un-letterboxes.
- `normalize_in_net` with mean=127/std=128 + `padding_color=127`: the app
  letterboxes with gray (127) padding and feeds raw uint8 RGB pixels — no
  manual normalization.
- HailoRT returns the NMS vstream as a ragged per-class list (NMS-by-score); the
  parser handles that plus object/dense layouts. First inference logs the raw
  type/shape for on-device verification (SOP §10).
- Class mapping: `cls_id` (0..88) → COCO category ID `cls_id+1`
  (`labels_offset=1`). The 10 unused COCO IDs are "N/A" and not drawn.

## Model source

The Hailo-8 HEF comes from the
[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst)
(model `efficientdet_lite2`).

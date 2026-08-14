# SSD MobileNet V1 on Raspberry Pi 5 / CM5 + Hailo-8

This module runs SSD MobileNet V1 object detection (COCO 80 thing classes,
90 category slots) with **on-chip NMS** (Hailo HPP). The classic TF SSD
detector (MobileNet V1 backbone) with on-chip NMS and predefined anchors. The
FastAPI service supports images, video files, USB cameras, an MJPEG preview,
and REST prediction endpoints.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-8 PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 4.23.x |
| Python | 3.11, aarch64 |
| Input | 300x300x3 RGB (normalize_in_net mean=127.5/std=127.5) |
| Output | on-chip NMS tensor, post-NMS shape 90x8x1 |
| Classes | 90 slots (COCO category IDs 1..90 via labels_offset=1; 10 unused IDs) |
| Parameters | 6.79M |
| Operations | 2.5G |
| HEF | Model Zoo v2.19.0, Hailo-8 |

The host driver, firmware, `libhailort.so`, and Python wheel must use the same
HailoRT major/minor version.

## Build

From the repository root:

```bash
sudo docker build -f docker/hailo8/ssd_mobilenet_v1.dockerfile \
  -t ssd_mobilenet_v1:latest \
  src/rpi5_hailo8_ssd_mobilenet_v1
```

## Run the demo video

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ssd_mobilenet_v1:latest \
  python web_detection.py --model_path model/ssd_mobilenet_v1.hef --video_path video/test.mp4
```

Open `http://<PI_IP>:8000`. For a USB camera, mount `/dev/video0` and replace
`--video_path video/test.mp4` with `--camera_id 0`.

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/ssd_mobilenet_v1/predict" \
  -F "file=@test.jpg"
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/api/models/ssd_mobilenet_v1/predict` | POST | Detections (JSON) |
| `/api/video_feed` | GET | MJPEG preview stream |

## Implementation notes

- The HEF runs NMS on-chip (`device_pre_post_layers: nms=true`); the app only
  parses the post-NMS tensor (per the official `tf_postproc_nms`), so
  `nms_thresh` is ignored (kept for API parity).
- Post-NMS rows are `[ymin, xmin, ymax, xmax, score, ...]`, normalized to [0,1]
  of the 300x300 letterboxed input; the app scales to pixels and un-letterboxes.
- `normalize_in_net` with mean=127.5/std=127.5 (classic SSD normalization
  `(pixel-127.5)/127.5`); the app feeds raw uint8 RGB pixels after letterboxing
  — no manual normalization.
- HailoRT returns the NMS vstream as a ragged per-class list (NMS-by-score); the
  parser handles that plus object/dense layouts. First inference logs the raw
  type/shape for on-device verification (SOP §10).
- Class mapping: `cls_id` (0..89) → COCO category ID `cls_id+1`
  (`labels_offset=1`). The 10 unused COCO IDs are "N/A" and not drawn.

## Model source

The Hailo-8 HEF comes from the
[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst)
(model `ssd_mobilenet_v1`).

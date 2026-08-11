# DAMO-YOLO (tinynasL25_S) on Raspberry Pi 5 / CM5 + Hailo-8

This module runs DAMO-YOLO object detection (TinyNAS-L25 backbone, Small size,
COCO 80 classes) with the `nanodet_split` post-processing head (DFL box
regression). It is the higher-accuracy Small (L25_S) variant of the damoyolo
family; same input/output contract as the L20_T build, deeper backbone. The
FastAPI service supports images, video files, USB cameras, an MJPEG preview,
and REST prediction endpoints.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-8 PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 4.23.x |
| Python | 3.11, aarch64 |
| Input | 640x640x3 RGB (normalization is a no-op; feed raw uint8) |
| Output | 6 heads: box (80x80x68, 40x40x68, 20x20x68) + cls (80x80x81, 40x40x81, 20x20x81) |
| Classes | 80 (COCO) |
| Parameters | 16.25M |
| Operations | 37.64G |
| HEF | Model Zoo v2.19.0, Hailo-8 |

The host driver, firmware, `libhailort.so`, and Python wheel must use the same
HailoRT major/minor version.

## Build

From the repository root:

```bash
sudo docker build -f docker/hailo8/damoyolo_tinynas_l25_s.dockerfile \
  -t damoyolo_tinynas_l25_s:latest \
  src/rpi5_hailo8_damoyolo_tinynas_l25_s
```

## Run the demo video

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  damoyolo_tinynas_l25_s:latest \
  python web_detection.py --model_path model/damoyolo_tinynas_l25_s.hef --video_path video/test.mp4
```

Open `http://<PI_IP>:8000`. For a USB camera, mount `/dev/video0` and replace
`--video_path video/test.mp4` with `--camera_id 0`.

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/damoyolo_tinynas_l25_s/predict" \
  -F "file=@test.jpg"
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/api/models/damoyolo_tinynas_l25_s/predict` | POST | Detections (JSON) |
| `/api/video_feed` | GET | MJPEG preview stream |

## Implementation notes

- Identical input/output contract and post-processing to the L20_T build; only
  the backbone differs (TinyNAS-L25, Small size → more accurate, heavier).
- The HEF uses `normalize_in_net` with mean=0/std=1 (a no-op) and
  `padding_color=0`; the app letterboxes to 640x640, converts BGR to RGB, and
  feeds raw uint8 pixels — no manual normalization.
- Only the classification sigmoid is on-chip (`device_pre_post_layers.sigmoid`).
  The grid decode, DFL box reduction (regression_length=16 → 4×17=68 channels),
  and per-class NMS run on the CPU, following the official Model Zoo
  `nanodet.py` (split_decode + _box_decoding).
- First inference prints every output vstream name/shape and the resolved
  per-scale box/cls grouping, so the head assignment can be verified on hardware.
- Defaults: confidence 0.25, IOU 0.45 (cleaner live preview). The Model Zoo
  eval uses 0.05 / 0.7 — lower the sliders to inspect lower-confidence boxes.

## Model source

The Hailo-8 HEF comes from the
[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst)
(model `damoyolo_tinynasL25_S`).

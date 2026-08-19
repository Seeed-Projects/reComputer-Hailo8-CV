# YOLOX-Tiny on Raspberry Pi 5 / CM5 + Hailo-8

This module runs YOLOX-Tiny object detection (COCO 80 classes, Megvii) with
**on-chip NMS** (Hailo HPP, `meta_arch=yolox`). The HEF performs NMS on-device
and emits already-decoded detections. The FastAPI service supports images,
video files, USB cameras, an MJPEG preview, and REST prediction endpoints.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-8 PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 4.23.x |
| Python | 3.11, aarch64 |
| Input | 416x416x3 RGB (normalize_in_net ImageNet RGB mean/std) |
| Output | on-chip NMS tensor, post-NMS shape 80x5x100 |
| Classes | 80 (COCO, 0-indexed) |
| Parameters | 5.05M |
| Operations | 6.44G |
| HEF | Model Zoo v2.19.0, Hailo-8 |

## Build

```bash
sudo docker build -f docker/hailo8/yolox_tiny.dockerfile \
  -t yolox_tiny:latest \
  src/rpi5_hailo8_yolox_tiny
```

## Run

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  yolox_tiny:latest \
  python web_detection.py --model_path model/yolox_tiny.hef --video_path video/test.mp4
```

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/yolox_tiny/predict" \
  -F "file=@test.jpg"
```

| Endpoint | Method | Description |
|---|---|---|
| `/api/models/yolox_tiny/predict` | POST | Detections (JSON) |
| `/api/video_feed` | GET | MJPEG preview stream |

## Implementation notes

- On-chip NMS (HPP, `meta_arch=yolox`); app only parses post-NMS tensor.
- `normalize_in_net` ImageNet RGB; no input_conversion → feed raw uint8 RGB.
- `padding_color=114` (gray, YOLOX convention).
- HailoRT ragged NMS-by-score output handled by the parser.
- Class mapping: cls_id (0..79) → standard COCO 80-class list directly.

## Model source

[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst)
(model `yolox_tiny`, source: [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)).

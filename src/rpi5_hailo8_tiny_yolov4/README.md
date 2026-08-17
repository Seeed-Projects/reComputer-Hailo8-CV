# Tiny-YOLOv4 on Raspberry Pi 5 / CM5 + Hailo-8

This module runs Tiny-YOLOv4 object detection (COCO 80 classes) with **CPU-side
YOLOv3 decode** (no on-chip NMS). Same architecture and I/O as Tiny-YOLOv3 but
with a YOLOv4-tiny backbone (lighter, more accurate). The HEF outputs two raw
heads; the app does the full grid decode (sigmoid + anchor + NMS) on the CPU.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-8 PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 4.23.x |
| Python | 3.11, aarch64 |
| Input | 416x416x3 RGB (normalize_in_net std=255) |
| Output | 2 raw heads: 13x13x255 (stride 32) + 26x26x255 (stride 16) |
| Classes | 80 (COCO, 0-indexed) |
| Parameters | 6.05M |
| Operations | 6.92G |
| HEF | Model Zoo v2.19.0, Hailo-8 |

## Build

```bash
sudo docker build -f docker/hailo8/tiny_yolov4.dockerfile \
  -t tiny_yolov4:latest \
  src/rpi5_hailo8_tiny_yolov4
```

## Run the demo video

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  tiny_yolov4:latest \
  python web_detection.py --model_path model/tiny_yolov4.hef --video_path video/test.mp4
```

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/tiny_yolov4/predict" \
  -F "file=@test.jpg"
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/api/models/tiny_yolov4/predict` | POST | Detections (JSON) |
| `/api/video_feed` | GET | MJPEG preview stream |

## Implementation notes

- Identical I/O contract and post-processing to Tiny-YOLOv3 (same base/yolo.yaml,
  same anchors, same YOLOv3 CPU decode); only the backbone differs (YOLOv4-tiny).
- No on-chip NMS — raw YOLOv3 heads, CPU decode (sigmoid + anchor + NMS).
- `normalize_in_net` std=255 (÷255); `padding_color=114` (gray). Feeds raw
  uint8 RGB after letterboxing.

## Model source

[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst)
(model `tiny_yolov4`).

# YOLOv5n-seg - Instance Segmentation

YOLOv5n-seg (anchor-based instance segmentation, 1.99M params) on Hailo-8.

## Model

| Property | Value |
|----------|-------|
| Architecture | YOLOv5n-seg (3 anchors, strides 8/16/32) |
| Input | 640×640×3 RGB |
| Output | 80-class boxes + instance masks (proto 160x160x32) |
| Parameters | 1.99M |
| Hardware mAP | 22.9 (COCO, Hailo Model Zoo reference) |
| Format | HEF (Hailo-8) |

## Quick Start

Runtime baseline: Python 3.11 and HailoRT 4.23.0. Run the build command from the repository root.

```bash
docker build -t yolov5n_seg -f docker/hailo8/yolov5n_seg.dockerfile src/rpi5_hailo8_yolov5n_seg

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  yolov5n_seg
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web preview |
| `/api/video_feed` | GET | MJPEG stream |
| `/api/models/yolov5n_seg/predict` | POST | Boxes + instance masks (JSON) |

## Source

HEF from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo).

# YOLO26m-seg - Instance Segmentation

YOLO26m-seg (23.6M params) on Hailo-8.

## Model

| Property | Value |
|----------|-------|
| Architecture | YOLO26m-seg |
| Input | 640×640×3 RGB |
| HEF output | Bounding boxes + instance-mask tensors (COCO 80 classes) |
| Parameters | 23.6M |
| Format | HEF (Hailo-8) |

## Quick Start

Runtime baseline: Python 3.11 and HailoRT 4.23.0. Run the build command from the repository root.

```bash
docker build -t yolo26m-seg -f docker/hailo8/yolo26m_seg.dockerfile src/hailo8_yolo26m_seg

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  yolo26m-seg
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web preview |
| `/api/video_feed` | GET | MJPEG stream |
| `/api/models/yolo26m_seg/predict` | POST | Box-level detections (JSON) |

The current Web postprocessor exposes box-level detections. Instance-mask decoding still needs to be validated on the target hardware. The checked-in YOLO26m HEF is unexpectedly small for the documented model size, so validate the artifact before release.

## Source

HEF from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo).

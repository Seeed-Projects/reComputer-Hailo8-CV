# YOLO11m-seg - Instance Segmentation

YOLO11m-seg (22.4M params) on Hailo-8.

## Model

| Property | Value |
|----------|-------|
| Architecture | YOLO11m-seg (anchor-free, one2many head, strides 8/16/32) |
| Input | 640x640x3 RGB (letterbox pad 114) |
| HEF output | 10 raw tensors: 64-ch DFL box, 80-ch class score and 32-ch mask-coefficient heads per stride, plus a 160x160x32 mask prototype |
| Parameters | 22.4M |
| Operations | 123.4G |
| mAP-seg (float / Hailo-8) | 41.7 / 40.9 |
| FPS (batch 1 / 8) | 36.8 / 59.7 |
| HEF | Hailo Model Zoo v2.20.0, Hailo-8 (39,601,115 bytes) |
| License | AGPL-3.0 (Ultralytics) |

## Quick Start

Runtime baseline: Python 3.11 and HailoRT 4.23.0. Run the build command from the repository root.

```bash
docker build -t yolov11m_seg -f docker/hailo8/yolov11m_seg.dockerfile src/rpi5_hailo8_yolov11m_seg

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  yolov11m_seg
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web preview |
| `/api/video_feed` | GET | MJPEG stream |
| `/api/models/yolov11m_seg/predict` | POST | Box-level detections (JSON) |

The HEF exposes raw heads and no on-chip NMS: the host runs the DFL box
decode, the sigmoid + per-class NMS and the mask assembly (coefficients x
prototype, cropped to each box). Instance masks are rendered in the MJPEG
preview; the REST payload lists boxes, classes and confidences.

## Source

HEF from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) v2.20.0 (Hailo-8):

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/yolov11m_seg.hef
```

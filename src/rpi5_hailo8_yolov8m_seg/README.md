# YOLOv8m-seg - Instance Segmentation

YOLOv8m-seg (27.3M params) on Hailo-8.

## Model

| Property | Value |
|----------|-------|
| Architecture | YOLOv8m-seg (anchor-free, one2many head, strides 8/16/32) |
| Input | 640x640x3 RGB (letterbox pad 114) |
| HEF output | 10 raw tensors: 64-ch DFL box, 80-ch class score and 32-ch mask-coefficient heads per stride, plus a 160x160x32 mask prototype |
| Parameters | 27.3M |
| Operations | 110.2G |
| Reference mAP | 40.6 (COCO instance segmentation, full precision, Hailo Model Zoo) |
| License | AGPL-3.0 (Ultralytics) |
| Format | HEF (Hailo-8) |

## Quick Start

Runtime baseline: Python 3.11 and HailoRT 4.23.0. Run the build command from the repository root.

```bash
docker build -t yolov8m_seg -f docker/hailo8/yolov8m_seg.dockerfile src/rpi5_hailo8_yolov8m_seg

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  yolov8m_seg
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web preview |
| `/api/video_feed` | GET | MJPEG stream |
| `/api/models/yolov8m_seg/predict` | POST | Box-level detections (JSON) |

The HEF exposes raw heads and no on-chip NMS (unlike the yolov8 detection
compile): the host runs the DFL box decode, the sigmoid + per-class NMS and
the mask assembly (coefficients x prototype, cropped to each box). Instance
masks are rendered in the MJPEG preview; the REST payload lists boxes,
classes and confidences.

## Source

HEF from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) v2.19.0 (Hailo-8):

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/yolov8m_seg.hef
```

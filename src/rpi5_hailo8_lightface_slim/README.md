# LightFace Slim - Face Detection

Ultra-lightweight face detection (0.26M params) on Hailo-8.

## Model

| Property | Value |
|----------|-------|
| Architecture | Ultra-Light-Fast-Generic-Face-Detector-1MB (SSD-style, strides 8/16/32/64) |
| Input | 240x320x3 uint8 BGR (aspect-preserving resize + bottom/right pad 0; mean 127 / std 128 baked into the HEF) |
| Output | 8 raw heads: 4 scales x {bbox deltas, background + face logits}; 4,420 anchors |
| Decode | Host-side SSD decode (variances 10/5) + softmax + greedy NMS (`meta_arch=retinaface`) |
| Parameters | 0.26M |
| Operations | 0.16G |
| Reference mAP | 39.71 (WIDER FACE, full precision, Hailo Model Zoo) |
| License | MIT (Ultra-Light-Fast-Generic-Face-Detector-1MB) |
| Format | HEF (Hailo-8) |

## Quick Start

Runtime baseline: Python 3.11 and HailoRT 4.23.0. Run the build command from the repository root.

```bash
docker build -t lightface_slim -f docker/hailo8/lightface_slim.dockerfile src/rpi5_hailo8_lightface_slim

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  lightface_slim
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web preview |
| `/api/video_feed` | GET | MJPEG stream |
| `/api/models/lightface_slim/predict` | POST | Face boxes + confidence (JSON) |

The HEF exposes raw heads and no on-chip NMS: the host runs the SSD decode
(variances 10/5), the softmax over the background/face logits, the score cut
and greedy NMS. The first inference logs every output vstream name, shape and
value range, so the layout in use is visible in the container log.

## Source

HEF from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) v2.19.0 (Hailo-8):

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/lightface_slim.hef
```

# YOLO26s-seg - Instance Segmentation

YOLO26s-seg (10.4M params) on Hailo-8. The module decodes the raw
segmentation heads, assembles the instance masks (mask coefficients x prototype,
cropped to each box) and draws them into the annotated preview.

## Model

| Property | Value |
|----------|-------|
| Architecture | YOLO26s-seg |
| Input | 640x640x3 RGB |
| HEF output | 10 raw tensors: 4-ch box (l,t / r,b distances), 80-ch class logits and 32-ch mask coefficients per stride, plus a 160x160x32 prototype |
| Parameters | 10.4M |
| Operations | 37G |
| Format | HEF (Hailo-8) |
| HEF size | 23,424,755 bytes |
| HEF SHA256 | `aa49b1c247a4d75ad221c51d43c3f740af1715b026717f871a50976e5a37be66` |

## Quick Start

Runtime baseline: Python 3.11 and HailoRT 4.23.0. Run the build command from
the repository root.

```bash
docker build -t yolo26s-seg -f docker/hailo8/yolo26s_seg.dockerfile src/rpi5_hailo8_yolo26s_seg

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo8-cv/yolo26s_seg:latest
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web preview |
| `/api/video_feed` | GET | MJPEG stream |
| `/api/models/yolo26s_seg/predict` | POST | Detections (JSON: class, confidence, box) |

The HEF exposes raw heads (no on-chip NMS): the host applies sigmoid, the
two-stage top-k of the one2one head (post_nms_topk=100, no NMS), the
regression_length=1 box decode, and the mask assembly. The JSON payload carries
the boxes; the instance masks are composited into the annotated preview and the
offline video output.

## Source

HEF from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) v2.19.0 (Hailo-8):

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/yolo26s_seg.hef
```

The Hailo-8 zoo compiles yolo26s_seg / yolo26m_seg / yolo26l_seg / yolo26x_seg (no n_seg build exists).

# RetinaFace MobileNet-v1 on Raspberry Pi 5 / CM5 + Hailo-8

This module runs RetinaFace MobileNet-v1 face detection (single "face" class,
WIDER FACE) with **CPU-decoded anchors** (`meta_arch=retinaface`). The HEF
exposes 9 raw heads — 3 strides (8/16/32), each with bbox, confidence and
5-point landmark predictions — and the app performs the full decode chain:
anchor generation (38,640 anchors), SSD-style box decode, softmax confidence,
greedy NMS, and landmark decode, ported from the Hailo Model Zoo.

The FastAPI service supports images, video files, USB cameras, an MJPEG
preview with box + keypoint overlays, and REST prediction endpoints.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-8 PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 4.23.x |
| Python | 3.11, aarch64 |
| Input | 736x1280x3 BGR (normalize_in_net mean [123,117,104], std 1) |
| Output | 9 heads: 3 scales x {bbox 8ch, conf 4ch, landmark 20ch} |
| Priors | 38,640 anchors (2/cell, feature maps 92x160/46x80/23x40) |
| Classes | 1 (face) + 5 landmarks per detection |
| Parameters | 3.49M |
| Operations | 25.14G |
| HEF | Model Zoo v2.19.0, Hailo-8 |

## Build

```bash
sudo docker build -f docker/hailo8/retinaface_mobilenet_v1.dockerfile \
  -t retinaface_mobilenet_v1:latest \
  src/rpi5_hailo8_retinaface_mobilenet_v1
```

## Run

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  retinaface_mobilenet_v1:latest \
  python web_detection.py --model_path model/retinaface_mobilenet_v1.hef --video_path video/test.mp4
```

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/retinaface_mobilenet_v1/predict" \
  -F "file=@test.jpg"
```

| Endpoint | Method | Description |
|---|---|---|
| `/api/models/retinaface_mobilenet_v1/predict` | POST | Faces + confidence + 5 landmarks (JSON) |
| `/api/video_feed` | GET | MJPEG preview stream (boxes + keypoints) |

## Implementation notes

- CPU post-processing (`meta_arch=retinaface`): no on-chip NMS. Anchor
  generation, SSD decode (variances 10/5), softmax conf, greedy NMS and
  landmark decode are a numpy port of the Model Zoo's
  `face_detection_postprocessing.py`.
- Head layout per official `collect_box_class_predictions`: the 8-channel
  head is bbox (2 anchors x 4 coords), the 4-channel head is confidence
  (2 anchors x {background, face}), the 20-channel head is landmarks
  (2 anchors x 10 coords).
- `normalize_in_net` with BGR mean [123, 117, 104] (std 1); the alls script
  applies `input_conversion(bgr_to_rgb)` in the HEF, so the app feeds raw
  uint8 **BGR** frames.
- Preprocessing: aspect-ratio-preserving resize + bottom/right pad (color 0)
  to 736x1280, matching the Model Zoo's `_ar_preserving_resize_and_crop`.
- Config defaults: `score_threshold=0.02` / `nms_iou_thresh=0.4` (official
  eval); the live preview starts at conf 0.5 for a cleaner image.
- The first-inference log prints every output vstream shape so the 9-head
  layout can be verified on hardware.

## Model source

[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_face_detection.rst)
(model `retinaface_mobilenet_v1`, source: [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)).
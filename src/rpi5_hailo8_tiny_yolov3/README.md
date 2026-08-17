# Tiny-YOLOv3 on Raspberry Pi 5 / CM5 + Hailo-8

This module runs Tiny-YOLOv3 object detection (COCO 80 classes) with **CPU-side
YOLOv3 decode** (no on-chip NMS). The HEF outputs two raw YOLOv3 heads; the app
does the full grid decode (sigmoid + anchor + NMS) on the CPU. The FastAPI
service supports images, video files, USB cameras, an MJPEG preview, and REST
prediction endpoints.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-8 PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 4.23.x |
| Python | 3.11, aarch64 |
| Input | 416x416x3 RGB (normalize_in_net std=255, ÷255) |
| Output | 2 raw heads: 13x13x255 (stride 32) + 26x26x255 (stride 16) |
| Classes | 80 (COCO, 0-indexed) |
| Parameters | 8.85M |
| Operations | 5.58G |
| HEF | Model Zoo v2.19.0, Hailo-8 |

## Build

From the repository root:

```bash
sudo docker build -f docker/hailo8/tiny_yolov3.dockerfile \
  -t tiny_yolov3:latest \
  src/rpi5_hailo8_tiny_yolov3
```

## Run the demo video

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  tiny_yolov3:latest \
  python web_detection.py --model_path model/tiny_yolov3.hef --video_path video/test.mp4
```

Open `http://<PI_IP>:8000`. For a USB camera, mount `/dev/video0` and replace
`--video_path video/test.mp4` with `--camera_id 0`.

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/tiny_yolov3/predict" \
  -F "file=@test.jpg"
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/api/models/tiny_yolov3/predict` | POST | Detections (JSON) |
| `/api/video_feed` | GET | MJPEG preview stream |

## Implementation notes

- **No on-chip NMS** — unlike the SSD/efficientdet/nanodet models, tiny_yolov3
  outputs raw YOLOv3 heads. The app does the full decode on CPU: sigmoid on
  obj + class scores, anchor-based grid decode, then NMS.
- Decode formula (from official `yolo.py` `_yolo3_decode`):
  `center = (sigmoid(raw_xy) + grid_offset) * stride`, `scale = exp(raw_wh) *
  anchor`, `score = sigmoid(obj) * sigmoid(cls)` (YOLOv3 obj × cls).
- `normalize_in_net` with std=255 (÷255); padding_color=114 (gray, YOLO
  convention). The app feeds raw uint8 RGB pixels after letterboxing — no manual
  normalization.
- 3 anchors per scale: stride 32 → [[81,82],[135,169],[344,319]], stride 16 →
  [[23,27],[37,58],[81,82]] (from the Model Zoo network YAML).
- First inference logs each head's shape for on-device verification (SOP §10).

## Model source

The Hailo-8 HEF comes from the
[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst)
(model `tiny_yolov3`).

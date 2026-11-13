# YOLOv3-Gluon on Raspberry Pi 5 / CM5 + Hailo-8

This module runs YOLOv3-Gluon object detection (COCO 80 classes) with **CPU-side
YOLOv3 decode** (no on-chip NMS). The HEF outputs three raw YOLOv3 heads; the
app does the full grid decode (sigmoid + anchor + NMS) on the CPU. The FastAPI
service supports images, video files, USB cameras, an MJPEG preview, and REST
prediction endpoints.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-8 PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 4.23.x |
| Python | 3.11, aarch64 |
| Input | 608x608x3 RGB (normalize_in_net; padding color 114) |
| Output | 3 raw heads: 19x19 / 38x38 / 76x76 (255 channels each) |
| Classes | 80 (COCO, 0-indexed) |
| Parameters | 68.79M |
| Operations | 140.7G |
| mAP (float / Hailo-8) | 37.3 / 35.8 |
| FPS (Hailo-8, batch 1) | 38.1 |
| HEF | Model Zoo v2.20.0, Hailo-8 (50,802,846 bytes) |

## Build

From the repository root:

```bash
sudo docker build -f docker/hailo8/yolov3_gluon.dockerfile \
  -t yolov3_gluon:latest \
  src/rpi5_hailo8_yolov3_gluon
```

## Run the demo video

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  yolov3_gluon:latest \
  python web_detection.py --model_path model/yolov3_gluon.hef --video_path video/test.mp4
```

Open `http://<BOARD_IP>:8000`. For a USB camera, mount `/dev/video0` and replace
`--video_path video/test.mp4` with `--camera_id 0`.

## REST API

```bash
curl -X POST "http://<BOARD_IP>:8000/api/models/yolov3_gluon/predict" \
  -F "file=@test.jpg"
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/api/models/yolov3_gluon/predict` | POST | Detections (JSON) |
| `/api/video_feed` | GET | MJPEG preview stream |

## Implementation notes

- **No on-chip NMS** — the HEF exposes three raw YOLOv3 heads. The app runs
  sigmoid, anchor-based grid decode and NMS on the CPU.
- Decode formula (from official `yolo.py` `_yolo3_decode`):
  `center = (sigmoid(raw_xy) + grid_offset) * stride`, `scale = exp(raw_wh) *
  anchor`, `score = sigmoid(obj) * sigmoid(cls)`.
- Letterbox with padding color 114, then BGR→RGB. The HEF applies the model
  normalization; feed raw uint8 RGB.
- Three anchor scales from the network YAML:
  stride 32 → [[116,90],[156,198],[373,326]], stride 16 →
  [[30,61],[62,45],[59,119]], stride 8 → [[10,13],[16,30],[33,23]].
- First inference logs each head's shape for on-device verification (SOP §10).

## Model source

The Hailo-8 HEF comes from
[Hailo Model Zoo v2.20.0](https://github.com/hailo-ai/hailo_model_zoo) (model
`yolov3_gluon`).

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/yolov3_gluon.hef
```

# STDC1 on Raspberry Pi 5 / CM5 + Hailo-8

This module provides real-time semantic segmentation with STDC1 and the 19
Cityscapes classes. The FastAPI service supports images, video files, USB
cameras, an MJPEG preview, and REST prediction endpoints.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-8 PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 4.23.x |
| Python | 3.11, aarch64 |
| Input | 1024x1920x3 RGB |
| Output | 19-class Cityscapes segmentation mask |
| Parameters | 8.27M |
| HEF | Model Zoo v2.19.0, Hailo-8 |

The host driver, firmware, `libhailort.so`, and Python wheel must use the same
HailoRT major/minor version.

## Build

From the repository root:

```bash
sudo docker build -f docker/hailo8/stdc1.dockerfile \
  -t stdc1:latest \
  src/rpi5_hailo8_stdc1
```

## Run the demo video

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  stdc1:latest \
  python web_detection.py --model_path model/stdc1.hef --video_path video/test.mp4
```

Open `http://<PI_IP>:8000`. For a USB camera, mount `/dev/video0` and replace
`--video_path video/test.mp4` with `--camera_id 0`.

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/stdc1/predict" \
  -F "file=@test.jpg"
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/api/models/stdc1/predict` | POST | Segmentation mask (JSON) |
| `/api/models/stdc1/visualize` | POST | Overlay image (JPEG) |
| `/api/models/stdc1/classes` | GET | Cityscapes class list |

## Model source

The Hailo-8 HEF comes from the
[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_semantic_segmentation.rst).

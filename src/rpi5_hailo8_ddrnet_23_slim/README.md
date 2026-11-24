# DDRNet-23-Slim on Raspberry Pi 5 / CM5 + Hailo-8

This module provides semantic segmentation with DDRNet-23-Slim and the 19
Cityscapes classes. The official Hailo-8 HEF takes a 1024x2048 RGB input, runs
normalization and argmax in-network, and returns a single-channel class-ID mask.
The FastAPI service supports images, video files, USB cameras, an MJPEG preview,
and REST prediction endpoints.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-8 PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 4.23.x |
| Python | 3.11, aarch64 |
| Input | 1024x2048x3 RGB |
| Output | 1024x2048x1 Cityscapes class-ID mask |
| Parameters | 5.70M |
| HEF | Model Zoo v2.20.0, Hailo-8 |

The host driver, firmware, `libhailort.so`, and Python wheel must use the same
HailoRT major/minor version.

## Build

From the repository root:

```bash
sudo docker build -f docker/hailo8/ddrnet_23_slim.dockerfile \
  -t ddrnet_23_slim:latest \
  src/rpi5_hailo8_ddrnet_23_slim
```

## Run the demo video

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ddrnet_23_slim:latest \
  python web_detection.py --model_path model/ddrnet_23_slim.hef --video_path video/test.mp4
```

Open `http://<PI_IP>:8000`. For a USB camera, mount `/dev/video0` and replace
`--video_path video/test.mp4` with `--camera_id 0`.

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/ddrnet_23_slim/predict" \
  -F "file=@test.jpg"
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/api/models/ddrnet_23_slim/predict` | POST | Segmentation mask (JSON) |
| `/api/models/ddrnet_23_slim/visualize` | POST | Overlay image (JPEG) |
| `/api/models/ddrnet_23_slim/classes` | GET | Cityscapes class list |

## Model source

The Hailo-8 HEF comes from the
[Hailo Model Zoo v2.20.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_semantic_segmentation.rst).
Official Hailo-8 figures are 76.7 float mIoU, 76.3 hardware mIoU, and 38.1 FPS
at batch size 1. These are Model Zoo measurements, not this WebUI's throughput.

The HEF applies normalization in-network (mean `[123.675, 116.28, 103.53]`,
std `[58.395, 57.12, 57.375]`, RGB order), so the application feeds raw RGB
uint8 pixels. The output is already an argmax class-ID mask; the host only maps
IDs to the Cityscapes palette.

See `TEST_REPORT.md` for the static checks and pending board checks.
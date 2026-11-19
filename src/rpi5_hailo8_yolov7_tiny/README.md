# YOLOv7-Tiny on Raspberry Pi 5 + Hailo-8

COCO 80-class object detection with the official Hailo-8 YOLOv7-Tiny HEF. The HEF
runs NMS on-chip; the application only parses the post-NMS result, draws boxes,
and serves the FastAPI/MJPEG WebUI.

## 1. Host preparation

```bash
sudo apt update
sudo apt install hailo-all
sudo reboot

hailortcli fw-control identify
ls /dev/hailo0

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh --mirror Aliyun
sudo systemctl enable docker
sudo systemctl start docker
```

The bundled wheel is `hailort-4.23.0-cp311-cp311-linux_aarch64.whl`; the host
firmware and `libhailort.so` must share the same major.minor.

## 2. Build and run

```bash
cd src/rpi5_hailo8_yolov7_tiny
sudo docker build -f ../../docker/hailo8/yolov7_tiny.dockerfile -t rpi5-hailo8-yolov7_tiny:latest .

sudo docker run --rm --privileged --net=host   -e PYTHONUNBUFFERED=1   --device /dev/hailo0:/dev/hailo0   -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro   -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro   rpi5-hailo8-yolov7_tiny:latest
```

The default command loops `video/test.mp4` and serves the WebUI on
`http://<Board_IP>:8000`. Replace `4.23.0` with the version reported by
`hailortcli fw-control identify`.

## 3. Model

| Item | Value |
|---|---|
| HEF | `model/yolov7_tiny.hef`, 7,373,091 bytes |
| SHA256 | `a0ae4b66e135d6e4c0b22aea7182c2ebd895cd23e1233e88d18309fa7734a177` |
| Input | 640x640x3 |
| Post-processing | On-chip NMS (HPP) |
| COCO mAP (float / hardware) | 37.1 / 36.3 |
| FPS (batch 1 / batch 8) | 633 / 633 |
| Parameters / operations | 6.22M / 13.74G |

## 4. HTTP API

- `GET /` — WebUI with MJPEG preview.
- `POST /api/models/yolov7/predict` — image or video-frame prediction.
- `POST /api/video/upload`, `POST /api/video/analyze`, `GET /api/video/status`
  — offline video analysis.
- `GET /api/video/download/{filename}` — annotated MP4 download.

## 5. Notes and limitations

- The HEF is the official Hailo-8 Model Zoo build; no hardware validation was
  possible in this development environment. See `TEST_REPORT.md`.
- NMS is inside the HEF; the IoU slider is display-only.
- The Web service targets a trusted LAN and has no authentication or TLS.
- Output videos have no audio.

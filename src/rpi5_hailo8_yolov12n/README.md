# YOLOv12n on Raspberry Pi 5 + Hailo-8

COCO 80-class object detection with the official Hailo-8 YOLOv12n HEF. The HEF
runs NMS on-chip; the application only parses the post-NMS result and serves the
FastAPI/MJPEG WebUI.

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
cd src/rpi5_hailo8_yolov12n
sudo docker build -f ../../docker/hailo8/yolov12n.dockerfile -t rpi5-hailo8-yolov12n:latest .

sudo docker run --rm --privileged --net=host   -e PYTHONUNBUFFERED=1   --device /dev/hailo0:/dev/hailo0   -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro   -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro   rpi5-hailo8-yolov12n:latest
```

The default command loops `video/test.mp4` and serves the WebUI on
`http://<Board_IP>:8000`.

## 3. Model

| Item | Value |
|---|---|
| HEF | `model/yolov12n.hef`, 8,940,675 bytes |
| SHA256 | `a1837b80fab4f102beb78c911cbfce2dcf93593d7015abecf2c1fadb2287d251` |
| Input | 640x640x3 |
| Post-processing | On-chip NMS (HPP) |
| COCO mAP (float / hardware) | 40.5 / 39.3 |
| FPS (batch 1 / batch 8) | 46.2 / 150 |
| Parameters / operations | 2.6M / 7.5G |

## 4. HTTP API

- `GET /` — WebUI with MJPEG preview.
- `POST /api/models/yolov12n/predict` — image or video-frame prediction.
- `POST /api/video/upload`, `POST /api/video/analyze`, `GET /api/video/status`
  — offline video analysis.
- `GET /api/video/download/{filename}` — annotated MP4 download.

## 5. Notes and limitations

- Upstream license is AGPL-3.0; review it before redistribution.
- The module has not been hardware-validated here. See `TEST_REPORT.md`.
- NMS is inside the HEF; the IoU slider is display-only.
- The Web service targets a trusted LAN and has no authentication or TLS.

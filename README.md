# reComputer-Hailo8-CV

[English] | [中文](./README_zh.md)

Industrial-grade computer vision reference for **Seeed reComputer series with Hailo-8 AI accelerator**.
YOLO object detection with real-time MJPEG preview, REST API, and offline batch
video analysis — designed for edge deployment with Hailo-8 PCIe/M.2 accelerators.

This repository serves as the **umbrella project** for Hailo-8 based CV solutions
across Seeed's reComputer product line. Platform-specific implementations are
organized in sub-repositories.

---

## Hardware platform

| | |
|---|---|
| Accelerator | Hailo-8 M.2 (PCIe), device node `/dev/hailo0` |
| Supported boards | Raspberry Pi 5 (reComputer R20 series), Jetson series, x86 industrial PCs |
| OS | Linux aarch64 / x86_64, kernel 5.15+ |
| Host drivers | `hailo-all` apt package (provides PCIe driver, firmware, `libhailort.so`) |
| HailoRT | 4.23.x validated — host driver / firmware / container wheel **must share major.minor** |

### Platform-specific implementations

| Platform | Repository | Status |
|---|---|---|
| **reComputer R20** (Raspberry Pi 5) | [reComputer-R20-CV](https://github.com/Seeed-Projects/reComputer-R20-CV) | ✅ Validated |
| reComputer Hailo-10H | [reComputer-Hailo10H-CV](https://github.com/Seeed-Projects/reComputer-Hailo10H-CV) | 🚧 In progress |

> **Looking for the Raspberry Pi 5 + Hailo-8 quick start?**
> Go to [reComputer-R20-CV](https://github.com/Seeed-Projects/reComputer-R20-CV)
> for the complete one-command Docker deployment with YOLOv8/v5/v10/v11 and more.

---

## Quick start (one-command, pre-built image)

The quick-start example below uses the published YOLOv8 image, which already
contains the source code, HailoRT wheel, ffmpeg, and the three Model Zoo `.hef`
weights (`yolov8n/s/m`). You only need a working Hailo toolchain on the host.

Published GHCR images are stored under this repository namespace:

| Model | Image |
|---|---|
| YOLOv8 | `ghcr.io/seeed-projects/recomputer-r20-cv/yolov8:latest` |
| YOLOv5 | `ghcr.io/seeed-projects/recomputer-r20-cv/yolov5:latest` |
| YOLOv10 | `ghcr.io/seeed-projects/recomputer-r20-cv/yolov10:latest` |
| YOLOv11 | `ghcr.io/seeed-projects/recomputer-r20-cv/yolov11:latest` |
| YOLOv8 Pose | `ghcr.io/seeed-projects/recomputer-r20-cv/yolov8_pose:latest` |
| SCRFD | `ghcr.io/seeed-projects/recomputer-r20-cv/scrfd:latest` |
| SCDepthV3 | `ghcr.io/seeed-projects/recomputer-r20-cv/scdepthv3:latest` |
| FastDepth | `ghcr.io/seeed-projects/recomputer-r20-cv/fast_depth:latest` |
| Person Attribute ResNet | `ghcr.io/seeed-projects/recomputer-r20-cv/person_attr_resnet:latest` |
| SegFormer B0 BN | `ghcr.io/seeed-projects/recomputer-r20-cv/segformer_b0_bn:latest` |
| U-Net MobileNetV2 | `ghcr.io/seeed-projects/recomputer-r20-cv/unet_mobilenet_v2:latest` |
| DeepLabV3 MobileNetV2 | `ghcr.io/seeed-projects/recomputer-r20-cv/deeplab_v3_mobilenet_v2:latest` |

### 1. Host prep (one-time, on the device)

#### Install Docker

Run the following commands on the development board to install Docker:

```bash
# Download installation script
curl -fsSL https://get.docker.com -o get-docker.sh
# Install using Aliyun mirror source
sudo sh get-docker.sh --mirror Aliyun
# Start Docker and enable auto-start on boot
sudo systemctl enable docker
sudo systemctl start docker
```

#### Install Hailo toolchain

```bash
sudo apt update
sudo apt install hailo-all
sudo reboot

# After reboot, confirm the chip and note the firmware version
hailortcli fw-control identify     # should report 4.23.0
ls /dev/hailo0
```

### 2. Run

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    ghcr.io/seeed-projects/recomputer-r20-cv/yolov8:latest
```

Docker will pull the image on first run (~1.8 GB). The container then loops the
bundled `video/test.mp4` and serves the Web UI on port `8000` — open
`http://<device_IP>:8000` in a browser.

> **Why the `libhailort.so` bind-mount?** The image ships only the Python
> bindings; the native library has to come from the host's `hailo-all` package.
> If your firmware version isn't `4.23.0`, replace both `4.23.0` references with
> the version printed by `hailortcli fw-control identify` (and rebuild the image
> from source against a matching wheel if the major.minor differs).

### USB camera mode

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    --device /dev/video0:/dev/video0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    ghcr.io/seeed-projects/recomputer-r20-cv/yolov8:latest \
    python web_detection.py --model_path model/yolov8n.hef --camera_id 0
```

---

## REST API

All endpoints listen on port `8000` of the container; with `--net=host` they're
reachable at `http://<device_IP>:8000`.

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/models/yolov8/predict` | POST | One-shot inference on uploaded image, specific video frame, or current camera frame |
| `/api/video_feed` | GET | MJPEG live stream with boxes overlaid (embed in an `<img>`) |
| `/api/config` | GET / POST | Read or update `obj_thresh` / `nms_thresh` |
| `/api/video/upload` | POST | Upload a video for batch analysis |
| `/api/video/analyze` | POST | Start an offline analysis job (form-data `filename=...`) |
| `/api/video/status` | GET | Poll job progress |
| `/api/video/list` | GET | List uploaded sources and finished outputs |
| `/api/video/download/{filename}` | GET | Download an annotated output |

### Inference examples

```bash
# Image upload
curl -X POST http://<device_IP>:8000/api/models/yolov8/predict -F "file=@cat.jpg"

# Specific frame of an uploaded video (timestamp in seconds)
curl -X POST http://<device_IP>:8000/api/models/yolov8/predict \
    -F "video=@test.mp4" -F "timestamp=5.5"

# Current camera frame
curl -X POST http://<device_IP>:8000/api/models/yolov8/predict -F "realtime=true"

# Per-call threshold override
curl -X POST http://<device_IP>:8000/api/models/yolov8/predict \
    -F "file=@cat.jpg" -F "conf=0.5" -F "iou=0.4"
```

Response:

```json
{
  "success": true,
  "source": "uploaded image",
  "predictions": [
    {
      "class": "car",
      "confidence": 0.787,
      "box": { "x1": 2108, "y1": 1483, "x2": 2291, "y2": 1651 }
    }
  ],
  "image": { "width": 3840, "height": 2160 }
}
```

Embed the live stream in any HTML page:

```html
<img src="http://<device_IP>:8000/api/video_feed">
```

### Dynamic threshold update

```bash
# Read current
curl http://<device_IP>:8000/api/config
# {"obj_thresh":0.25,"nms_thresh":0.45}

# Update (either field is optional)
curl -X POST http://<device_IP>:8000/api/config \
     -H "Content-Type: application/json" \
     -d '{"obj_thresh":0.4}'
```

> `nms_thresh` is kept for API compatibility, but the Model Zoo `yolov8n.hef`
> performs NMS on-chip — the slider only acts as an extra confidence filter.

---

## Adapting to other models

Treat each platform's module as a **template**:

1. Copy the whole directory, rename (e.g. `rpi5_hailo8_yolov8_seg/`).
2. Drop the new `.hef` into `model/`.
3. If the model uses the **same output layout** (NMS on-chip,
   `(1, num_classes, max_dets, 5)`) — nothing else changes.
4. For seg / pose / obb / non-NMS models, rewrite `post_process_hailo()` in
   `web_detection.py` against the actual tensor spec. Hailo Model Zoo's per-model
   README documents the output layout.
5. Add a matching `docker/hailo8/<model>.dockerfile`.

Walkthrough for R20 platform:
[reComputer-R20-CV § Adapting to other models](https://github.com/Seeed-Projects/reComputer-R20-CV#adapting-to-other-models).

---

## Documentation

Platform-specific deep dives (deployment, CLI arguments, troubleshooting):

- [reComputer-R20-CV](https://github.com/Seeed-Projects/reComputer-R20-CV) —
  Raspberry Pi 5 + Hailo-8: full quick-start guide, module deep dive, validation reports
- [reComputer-Hailo10H-CV](https://github.com/Seeed-Projects/reComputer-Hailo10H-CV) —
  Hailo-10H platform (coming soon)

---

## License

MIT License

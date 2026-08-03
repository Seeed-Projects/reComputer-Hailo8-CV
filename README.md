# reComputer-Hailo8-CV

[English] | [中文](./README_zh.md)

Industrial-grade computer vision reference for **Seeed reComputer series with Hailo-8 AI accelerator**.
YOLO object detection with real-time MJPEG preview, REST API, and offline batch
video analysis — designed for edge deployment with Hailo-8 PCIe/M.2 accelerators.

This repository serves as the **umbrella project** for Hailo-8 based CV solutions
across Seeed's reComputer product line. Platform-specific implementations are
organized in sub-repositories.

---

## Supported platforms

| Platform | Repository | Accelerator | Status |
|---|---|---|---|
| **reComputer R20** (Raspberry Pi 5) | [reComputer-R20-CV](https://github.com/Seeed-Projects/reComputer-R20-CV) | Hailo-8 M.2 (PCIe) | ✅ Validated |
| reComputer Hailo-10H | [reComputer-Hailo10H-CV](https://github.com/Seeed-Projects/reComputer-Hailo10H-CV) | Hailo-10H | 🚧 In progress |

> **Looking for the Raspberry Pi 5 + Hailo-8 quick start?**
> Go to [reComputer-R20-CV](https://github.com/Seeed-Projects/reComputer-R20-CV)
> for the complete one-command Docker deployment with YOLOv8/v5/v10/v11 and more.

---

## Architecture overview

All platform implementations share a common architecture:

```
┌─────────────────────────────────────────────────┐
│  Host OS (Linux aarch64)                        │
│  ├── HailoRT driver + firmware (hailo-all)      │
│  ├── Docker Engine                              │
│  └── /dev/hailo0  (PCIe device node)            │
├─────────────────────────────────────────────────┤
│  Container (python:3.11-slim arm64)             │
│  ├── HailoRT Python wheel                       │
│  ├── FastAPI web server (port 8000)             │
│  ├── Inference pipeline (HailoRT InferVStreams) │
│  ├── MJPEG encoder + video processing           │
│  └── Model weights (.hef)                       │
└─────────────────────────────────────────────────┘
```

### Common design principles

- **Host-driver / container-wheel version alignment**: The `libhailort.so` native
  library is bind-mounted from the host to ensure the container's Python bindings
  match the host's PCIe driver and firmware.
- **No hardware video encoder dependency**: The pipeline uses MJPEG software
  encoding, avoiding platform-specific V4L2 encoder quirks.
- **Template-based module design**: Each model (YOLOv8, YOLOv5, etc.) is a
  self-contained module — copy it, swap the `.hef`, and adjust post-processing.

---

## Quick start

### For reComputer R20 (Raspberry Pi 5 + Hailo-8)

See the full guide at [reComputer-R20-CV](https://github.com/Seeed-Projects/reComputer-R20-CV).

One-command summary:

```bash
# 1. Install Docker and Hailo toolchain on the Pi
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh --mirror Aliyun
sudo systemctl enable --now docker
sudo apt update && sudo apt install hailo-all && sudo reboot

# 2. Run YOLOv8 detection
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    ghcr.io/seeed-projects/recomputer-r20-cv/yolov8:latest
```

Open `http://<device_ip>:8000` in a browser for the Web UI.

### Available Docker images

All pre-built images are published to GHCR under the
`ghcr.io/seeed-projects/recomputer-r20-cv/` namespace:

| Model | Image |
|---|---|
| YOLOv8 | `ghcr.io/seeed-projects/recomputer-r20-cv/yolov8:latest` |
| YOLOv5 | `ghcr.io/seeed-projects/recomputer-r20-cv/yolov5:latest` |
| YOLOv10 | `ghcr.io/seeed-projects/recomputer-r20-cv/yolov10:latest` |
| YOLOv11 | `ghcr.io/seeed-projects/recomputer-r20-cv/yolov11:latest` |
| YOLOv8 Pose | `ghcr.io/seeed-projects/recomputer-r20-cv/yolov8_pose:latest` |
| SCRFD (Face Detection) | `ghcr.io/seeed-projects/recomputer-r20-cv/scrfd:latest` |
| SCDepthV3 (Depth) | `ghcr.io/seeed-projects/recomputer-r20-cv/scdepthv3:latest` |
| FastDepth (Depth) | `ghcr.io/seeed-projects/recomputer-r20-cv/fast_depth:latest` |
| Person Attribute ResNet | `ghcr.io/seeed-projects/recomputer-r20-cv/person_attr_resnet:latest` |
| SegFormer B0 BN | `ghcr.io/seeed-projects/recomputer-r20-cv/segformer_b0_bn:latest` |
| U-Net MobileNetV2 | `ghcr.io/seeed-projects/recomputer-r20-cv/unet_mobilenet_v2:latest` |
| DeepLabV3 MobileNetV2 | `ghcr.io/seeed-projects/recomputer-r20-cv/deeplab_v3_mobilenet_v2:latest` |

---

## REST API reference

All platform implementations expose a consistent REST API on port `8000`:

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/models/{model}/predict` | POST | One-shot inference (image / video frame / camera) |
| `/api/video_feed` | GET | MJPEG live stream with detection overlays |
| `/api/config` | GET / POST | Read or update detection thresholds |
| `/api/video/upload` | POST | Upload video for batch analysis |
| `/api/video/analyze` | POST | Start offline analysis job |
| `/api/video/status` | GET | Poll job progress |
| `/api/video/list` | GET | List uploaded sources and outputs |
| `/api/video/download/{filename}` | GET | Download annotated output |

Interactive API docs available at `http://<device_ip>:8000/docs`.

---

## Adapting to other models

The module design is intentionally **template-based**:

1. Copy an existing module directory (e.g., `rpi5_hailo8_yolov8/`).
2. Drop the new `.hef` model into `model/`.
3. If the model uses the **same output layout** (NMS on-chip), no code changes needed.
4. For seg / pose / obb / non-NMS models, rewrite `post_process_hailo()` against
   the actual tensor spec (see Hailo Model Zoo documentation).
5. Add a matching Dockerfile.

Detailed walkthrough in each platform's sub-repository.

---

## Documentation

Platform-specific deep dives:

- [reComputer-R20-CV](https://github.com/Seeed-Projects/reComputer-R20-CV) —
  Raspberry Pi 5 + Hailo-8: deployment, CLI, troubleshooting, validation reports

---

## Hardware requirements

| Component | Minimum | Recommended |
|---|---|---|
| AI Accelerator | Hailo-8 M.2 (26 TOPS) | Hailo-8 / Hailo-10H |
| Interface | PCIe Gen3 x1 (M.2 M-key) | PCIe Gen3 x2+ |
| Host RAM | 4 GB | 8 GB+ |
| Storage | 16 GB SD/eMMC | 32 GB+ SSD |
| OS | Linux aarch64, kernel 5.15+ | Raspberry Pi OS Bookworm / Ubuntu 22.04+ |

---

## License

MIT License

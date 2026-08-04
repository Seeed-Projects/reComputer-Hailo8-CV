# reComputer-Hailo8-CV

[English] | [中文](./README_zh.md)

Hailo-8 computer-vision applications for Raspberry Pi 5 / CM5-based Seeed
reComputer R Series devices, using HailoRT 4.23.x.

## Included models

| Model | Task | Parameters | Module | Container image |
|---|---|---:|---|---|
| CenterPose RegNetX-800MF | Multi-person pose estimation | 12.31M | `src/rpi5_hailo8_centerpose_regnetx_800mf/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/centerpose_regnetx_800mf:latest` |
| STDC1 | Semantic segmentation | 8.27M | `src/rpi5_hailo8_stdc1/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/stdc1:latest` |

## Repository layout

```text
reComputer-Hailo8-CV/
├── .github/workflows/build-ghcr-images.yml
├── docker/hailo8/
│   ├── centerpose_regnetx_800mf.dockerfile
│   └── stdc1.dockerfile
└── src/
    ├── rpi5_hailo8_centerpose_regnetx_800mf/
    └── rpi5_hailo8_stdc1/
```

## Run STDC1

The host must expose `/dev/hailo0` and use HailoRT 4.23.x.

```bash
sudo docker run --rm \
  --name pi5-hailo8-stdc1 \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo8-cv/stdc1:latest \
  python web_detection.py --model_path model/stdc1.hef --video_path video/test.mp4
```

Open `http://<device_IP>:8000`. See each module README for build, camera, and
REST API instructions.

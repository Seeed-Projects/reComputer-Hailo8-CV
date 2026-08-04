# reComputer-Hailo8-CV

[English](./README.md) | 中文

本仓库为基于 Raspberry Pi 5 / CM5 的 Seeed reComputer R Series 设备提供
Hailo-8 计算机视觉应用，运行时版本为 HailoRT 4.23.x。

## 已包含模型

| 模型 | 任务 | 参数量 | 模块 | 容器镜像 |
|---|---|---:|---|---|
| CenterPose RegNetX-800MF | 多人姿态估计 | 12.31M | `src/rpi5_hailo8_centerpose_regnetx_800mf/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/centerpose_regnetx_800mf:latest` |
| STDC1 | 语义分割 | 8.27M | `src/rpi5_hailo8_stdc1/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/stdc1:latest` |

## 仓库结构

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

## 运行 STDC1

宿主机必须存在 `/dev/hailo0`，并使用 HailoRT 4.23.x。

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

浏览器打开 `http://<设备IP>:8000`。构建、摄像头和 REST API 说明见各模块 README。

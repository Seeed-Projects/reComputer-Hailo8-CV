# reComputer-Hailo8-CV

[English](./README.md) | 中文

本仓库为 Seeed reComputer 设备提供 Hailo-8 计算机视觉应用。首个模块为
CenterPose RegNetX-800MF，在 Raspberry Pi 5 / CM5 与 HailoRT 4.23.x 上运行；
目录和 Docker 约定参考
[`reComputer-R20-CV`](https://github.com/Seeed-Projects/reComputer-R20-CV)。

## 已包含模型

| 模型 | 任务 | 参数量 | 模块 | 容器镜像 |
|---|---|---:|---|---|
| CenterPose RegNetX-800MF | 多人姿态估计 | 12.31M | `src/rpi5_hailo8_centerpose_regnetx_800mf/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/centerpose_regnetx_800mf:latest` |

## 仓库结构

```text
reComputer-Hailo8-CV/
├── .github/workflows/build-ghcr-images.yml
├── docker/hailo8/centerpose_regnetx_800mf.dockerfile
└── src/rpi5_hailo8_centerpose_regnetx_800mf/
    ├── hailort-packages/
    ├── model/centerpose_regnetx_800mf.hef
    ├── py_utils/
    ├── video/test.mp4
    ├── web_detection.py
    ├── requirements.txt
    ├── README.md
    └── README_zh.md
```

## 运行

宿主机需要存在 `/dev/hailo0`，并安装 HailoRT 4.23.x。

```bash
sudo docker run --rm \
  --name pi5-hailo8-centerpose \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo8-cv/centerpose_regnetx_800mf:latest \
  python web_detection.py \
    --model_path model/centerpose_regnetx_800mf.hef \
    --video_path video/test.mp4
```

浏览器打开 `http://<设备IP>:8000`。构建、摄像头和 REST API 说明见模块
[中文 README](./src/rpi5_hailo8_centerpose_regnetx_800mf/README_zh.md)。

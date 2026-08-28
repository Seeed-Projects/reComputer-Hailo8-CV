# reComputer-Hailo8-CV

[English](./README.md) | [中文]

面向 **Raspberry Pi 5 / CM5 + Hailo-8**（reComputer R 系列）的工业级计算机视觉
参考实现。每个模型是一个独立模块，提供 FastAPI 服务：实时 MJPEG 预览、REST
推理、USB 摄像头、离线批量视频分析——围绕 PCIe 接口的 Hailo-8 加速器和
HailoRT 4.23.x 构建。

仓库覆盖三类任务——**目标检测**（CenterNet、DAMO-YOLO、EfficientDet、
NanoDet、SSD、Tiny-YOLO）、**语义分割**（STDC1）、**姿态估计**（CenterPose）。
所有模块共用同一套骨架；部分模型使用片上 NMS（Hailo HPP，输出已解码框），
部分模型（Tiny-YOLOv3/v4）输出 raw head，需 CPU 端完整 YOLOv3 解码。

---

## 硬件平台

| | |
|---|---|
| 主板 | Raspberry Pi 5 / CM5（reComputer R 系列载板） |
| 加速器 | Hailo-8 M.2（PCIe），设备节点 `/dev/hailo0` |
| 系统 | Raspberry Pi OS Bookworm，内核 6.12+ aarch64 |
| 宿主驱动 | `hailort hailort-pcie-driver python3-hailort`（PCIe 驱动 + 固件 + Python API） |
| HailoRT | 4.23.x 已验证——驱动 / 固件 / 容器 wheel **主.次版本必须一致** |

---

## 已收录模型（22 个）

| 模型 | 任务 | 参数量 | 模块 | 容器镜像 |
|---|---|---:|---|---|
| CenterPose RegNetX-800MF | 姿态估计（17 关键点） | 12.31M | `src/rpi5_hailo8_centerpose_regnetx_800mf/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/centerpose_regnetx_800mf:latest` |
| STDC1 | 语义分割（Cityscapes 19） | 8.27M | `src/rpi5_hailo8_stdc1/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/stdc1:latest` |
| CenterNet (resnet_v1_18) | 目标检测（COCO 80） | 14.22M | `src/rpi5_hailo8_centernet_resnet_v1_18_postprocess/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/centernet_resnet_v1_18_postprocess:latest` |
| CenterNet (resnet_v1_50) | 目标检测（COCO 80） | 30.07M | `src/rpi5_hailo8_centernet_resnet_v1_50_postprocess/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/centernet_resnet_v1_50_postprocess:latest` |
| DAMO-YOLO (tinynasL20_T) | 目标检测（COCO 80） | 11.35M | `src/rpi5_hailo8_damoyolo_tinynas_l20_t/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/damoyolo_tinynas_l20_t:latest` |
| DAMO-YOLO (tinynasL25_S) | 目标检测（COCO 80） | 16.25M | `src/rpi5_hailo8_damoyolo_tinynas_l25_s/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/damoyolo_tinynas_l25_s:latest` |
| DAMO-YOLO (tinynasL35_M) | 目标检测（COCO 80） | 33.98M | `src/rpi5_hailo8_damoyolo_tinynas_l35_m/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/damoyolo_tinynas_l35_m:latest` |
| EfficientDet-Lite0 | 目标检测（COCO 80） | 3.56M | `src/rpi5_hailo8_efficientdet_lite0/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/efficientdet_lite0:latest` |
| EfficientDet-Lite1 | 目标检测（COCO 80） | 4.73M | `src/rpi5_hailo8_efficientdet_lite1/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/efficientdet_lite1:latest` |
| EfficientDet-Lite2 | 目标检测（COCO 80） | 5.93M | `src/rpi5_hailo8_efficientdet_lite2/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/efficientdet_lite2:latest` |
| NanoDet-RepVGG | 目标检测（COCO 80） | 6.74M | `src/rpi5_hailo8_nanodet_repvgg/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/nanodet_repvgg:latest` |
| NanoDet-RepVGG-a12 | 目标检测（COCO 80） | 5.13M | `src/rpi5_hailo8_nanodet_repvgg_a12/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/nanodet_repvgg_a12:latest` |
| NanoDet-RepVGG-a1-640 | 目标检测（COCO 80） | 10.79M | `src/rpi5_hailo8_nanodet_repvgg_a1_640/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/nanodet_repvgg_a1_640:latest` |
| SSD MobileNet V1 | 目标检测（COCO 80） | 6.79M | `src/rpi5_hailo8_ssd_mobilenet_v1/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/ssd_mobilenet_v1:latest` |
| SSD MobileNet V2 | 目标检测（COCO 80） | 4.46M | `src/rpi5_hailo8_ssd_mobilenet_v2/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/ssd_mobilenet_v2:latest` |
| Tiny-YOLOv3 | 目标检测（COCO 80） | 8.85M | `src/rpi5_hailo8_tiny_yolov3/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/tiny_yolov3:latest` |
| Tiny-YOLOv4 | 目标检测（COCO 80） | 6.05M | `src/rpi5_hailo8_tiny_yolov4/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/tiny_yolov4:latest` |
| Tiny-YOLOv4 License Plates | 车牌检测 | 5.87M | `src/rpi5_hailo8_tiny_yolov4_license_plates/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/tiny_yolov4_license_plates:latest` |
| LPRNet | 车牌检测 + 数字 OCR 流水线 | 7.14M OCR | `src/rpi5_hailo8_lprnet/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/lprnet:latest` |

HEF 均来自官方 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) 的 Hailo-8 构建。多数模型使用 v2.19.0；LPR 模型使用当前可用的 v2.16/v2.17 Hailo-8 产物，详见模块 README。

### 后处理架构

| 架构 | 模型 | 片上 NMS | 输出格式 |
|---|---|---|---|
| 片上 NMS（HPP） | EfficientDet、NanoDet、SSD | 是 | 后 NMS 张量（Cx5xD） |
| 片上 max_finder | CenterNet | 部分 | 稀疏热图（128x128xC） |
| CPU YOLOv3 解码 | Tiny-YOLOv3、Tiny-YOLOv4 | 否 | Raw head（HxWx255） |
| 车牌流水线 | Tiny-YOLOv4 License Plates、LPRNet | 否 | HxWx18 检测 head + 5x19x11 OCR logits |
| CPU DFL 解码 | DAMO-YOLO | 否 | Raw nanodet_split head |
| CPU 6-head 解码 | CenterPose | 否 | Raw CenterNet head + 关键点 |

---

## 快速开始

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    ghcr.io/seeed-projects/recomputer-hailo8-cv/damoyolo_tinynas_l25_s:latest
```

浏览器打开 `http://<设备_IP>:8000`。

### 宿主准备

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh --mirror Aliyun
sudo systemctl enable docker && sudo systemctl start docker

sudo apt update
sudo apt install hailort hailort-pcie-driver python3-hailort
sudo reboot
hailortcli fw-control identify
ls /dev/hailo0
```

> 直接装 `hailort hailort-pcie-driver python3-hailort`，不要用 `hailo-all`（可能拉到 Hailo-10H / 5.x 包，与 Hailo-8 4.23.x 不匹配）。

### USB 摄像头模式

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    --device /dev/video0:/dev/video0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    ghcr.io/seeed-projects/recomputer-hailo8-cv/damoyolo_tinynas_l25_s:latest \
    python web_detection.py --model_path model/damoyolo_tinynas_l25_s.hef --camera_id 0
```

---

## REST API

所有接口监听容器 `8000` 端口；用 `--net=host` 时可通过 `http://<设备_IP>:8000`
访问。把 `<slug>` 替换为上表中的模型 slug。

| 接口 | 方法 | 用途 |
|---|---|---|
| `/api/models/<slug>/predict` | POST | 对上传图片、指定视频帧或当前摄像头帧做单次推理 |
| `/api/video_feed` | GET | MJPEG 实时流（嵌入 `<img>`） |
| `/api/config` | GET / POST | 读取或更新 `obj_thresh` / `nms_thresh` |
| `/api/video/upload` | POST | 上传视频用于批量分析 |
| `/api/video/analyze` | POST | 启动离线分析任务 |
| `/api/video/status` | GET | 查询任务进度 |
| `/api/video/list` | GET | 列出已上传源文件和已完成输出 |
| `/api/video/download/{filename}` | GET | 下载标注后的输出 |

```bash
curl -X POST http://<设备_IP>:8000/api/models/damoyolo_tinynas_l25_s/predict -F "file=@test.jpg"
```

> 片上 NMS 模型的 `nms_thresh` 滑块无实际效果（NMS 已在设备上完成）；CPU 解码模型
> 的 `nms_thresh` 有效。

---

## 适配其他模型

1. 复制模块并改名（`rpi5_hailo8_<新_slug>/`）。
2. 把新 `.hef` 放入 `model/`（小写 slug 名）。
3. 添加 `docker/hailo8/<slug>.dockerfile` + CI matrix 项。
4. **按真实 HEF 输出重新推导后处理**——查 Model Zoo YAML 的输出布局，首次推理
   验证 RGB/BGR 和归一化（SOP §10）。
5. 更新 `README*.md` 和 `TEST_REPORT.md`。

完整清单：`docs/CM5_HAILO8_MODEL_DEVELOPMENT_SOP_zh.md`

---

## 文档

- [CenterPose RegNetX-800MF](src/rpi5_hailo8_centerpose_regnetx_800mf/README_zh.md) — [English](src/rpi5_hailo8_centerpose_regnetx_800mf/README.md)
- [STDC1](src/rpi5_hailo8_stdc1/README_zh.md) — [English](src/rpi5_hailo8_stdc1/README.md)
- [CenterNet (resnet_v1_18)](src/rpi5_hailo8_centernet_resnet_v1_18_postprocess/README_zh.md) — [English](src/rpi5_hailo8_centernet_resnet_v1_18_postprocess/README.md)
- [CenterNet (resnet_v1_50)](src/rpi5_hailo8_centernet_resnet_v1_50_postprocess/README_zh.md) — [English](src/rpi5_hailo8_centernet_resnet_v1_50_postprocess/README.md)
- [DAMO-YOLO (tinynasL20_T)](src/rpi5_hailo8_damoyolo_tinynas_l20_t/README_zh.md) — [English](src/rpi5_hailo8_damoyolo_tinynas_l20_t/README.md)
- [DAMO-YOLO (tinynasL25_S)](src/rpi5_hailo8_damoyolo_tinynas_l25_s/README_zh.md) — [English](src/rpi5_hailo8_damoyolo_tinynas_l25_s/README.md)
- [DAMO-YOLO (tinynasL35_M)](src/rpi5_hailo8_damoyolo_tinynas_l35_m/README_zh.md) — [English](src/rpi5_hailo8_damoyolo_tinynas_l35_m/README.md)
- [EfficientDet-Lite0](src/rpi5_hailo8_efficientdet_lite0/README_zh.md) — [English](src/rpi5_hailo8_efficientdet_lite0/README.md)
- [EfficientDet-Lite1](src/rpi5_hailo8_efficientdet_lite1/README_zh.md) — [English](src/rpi5_hailo8_efficientdet_lite1/README.md)
- [EfficientDet-Lite2](src/rpi5_hailo8_efficientdet_lite2/README_zh.md) — [English](src/rpi5_hailo8_efficientdet_lite2/README.md)
- [NanoDet-RepVGG](src/rpi5_hailo8_nanodet_repvgg/README_zh.md) — [English](src/rpi5_hailo8_nanodet_repvgg/README.md)
- [NanoDet-RepVGG-a12](src/rpi5_hailo8_nanodet_repvgg_a12/README_zh.md) — [English](src/rpi5_hailo8_nanodet_repvgg_a12/README.md)
- [NanoDet-RepVGG-a1-640](src/rpi5_hailo8_nanodet_repvgg_a1_640/README_zh.md) — [English](src/rpi5_hailo8_nanodet_repvgg_a1_640/README.md)
- [SSD MobileNet V1](src/rpi5_hailo8_ssd_mobilenet_v1/README_zh.md) — [English](src/rpi5_hailo8_ssd_mobilenet_v1/README.md)
- [SSD MobileNet V2](src/rpi5_hailo8_ssd_mobilenet_v2/README_zh.md) — [English](src/rpi5_hailo8_ssd_mobilenet_v2/README.md)
- [Tiny-YOLOv3](src/rpi5_hailo8_tiny_yolov3/README_zh.md) — [English](src/rpi5_hailo8_tiny_yolov3/README.md)
- [Tiny-YOLOv4](src/rpi5_hailo8_tiny_yolov4/README_zh.md) — [English](src/rpi5_hailo8_tiny_yolov4/README.md)
- [Tiny-YOLOv4 License Plates](src/rpi5_hailo8_tiny_yolov4_license_plates/README_zh.md) — [English](src/rpi5_hailo8_tiny_yolov4_license_plates/README.md)
- [LPRNet 流水线](src/rpi5_hailo8_lprnet/README_zh.md) — [English](src/rpi5_hailo8_lprnet/README.md)

每个模块带 `TEST_REPORT.md` 验证记录。

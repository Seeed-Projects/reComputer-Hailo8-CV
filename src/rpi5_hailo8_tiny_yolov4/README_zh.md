# Tiny-YOLOv4：Raspberry Pi 5 / CM5 + Hailo-8 目标检测

本模块运行 Tiny-YOLOv4 目标检测（COCO 80 类），使用 **CPU 端 YOLOv3 解码**
（无片上 NMS）。与 Tiny-YOLOv3 同架构同 I/O，仅主干换为 YOLOv4-tiny（更轻、
更准）。HEF 输出两个 raw head，应用在 CPU 上完成完整解码。

## 兼容环境

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-8 PCIe（`/dev/hailo0`） |
| 宿主机/运行时 | HailoRT 4.23.x |
| Python | 3.11，aarch64 |
| 输入 | 416x416x3 RGB（normalize_in_net std=255） |
| 输出 | 2 个 raw head：13x13x255 (stride 32) + 26x26x255 (stride 16) |
| 类别 | 80（COCO，0-indexed） |
| 参数量 | 6.05M |
| 运算量 | 6.92G |
| HEF | Model Zoo v2.19.0，Hailo-8 |

## 构建

```bash
sudo docker build -f docker/hailo8/tiny_yolov4.dockerfile \
  -t tiny_yolov4:latest \
  src/rpi5_hailo8_tiny_yolov4
```

## 运行演示视频

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  tiny_yolov4:latest \
  python web_detection.py --model_path model/tiny_yolov4.hef --video_path video/test.mp4
```

浏览器打开 `http://<PI_IP>:8000`。USB 摄像头：挂载 `/dev/video0`，把
`--video_path video/test.mp4` 替换为 `--camera_id 0`。

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/tiny_yolov4/predict" \
  -F "file=@test.jpg"
```

| 接口 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 健康检查 |
| `/api/models/tiny_yolov4/predict` | POST | 检测结果（JSON） |
| `/api/video_feed` | GET | MJPEG 预览流 |

## 实现说明

- 与 Tiny-YOLOv3 同 I/O 契约和后处理（同 base/yolo.yaml、同 anchors、同
  YOLOv3 CPU 解码）；仅主干不同（YOLOv4-tiny）。
- 无片上 NMS——raw YOLOv3 head，CPU 解码（sigmoid + anchor + NMS）。
- `normalize_in_net`（std=255，÷255）+ `padding_color=114`（灰色）。
  应用在 letterbox 后直接喂原始 uint8 RGB 像素。

## 模型来源

[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst)
（模型 `tiny_yolov4`）。

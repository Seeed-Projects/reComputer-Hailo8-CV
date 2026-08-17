# Tiny-YOLOv3：Raspberry Pi 5 / CM5 + Hailo-8 目标检测

本模块运行 Tiny-YOLOv3 目标检测（COCO 80 类），使用**CPU 端 YOLOv3 解码**
（无片上 NMS）。HEF 输出两个 raw YOLOv3 head，应用在 CPU 上完成完整解码
（sigmoid + anchor + NMS）。FastAPI 服务支持图片、视频文件、USB 摄像头、
MJPEG 预览和 REST 推理接口。

## 兼容环境

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-8 PCIe（`/dev/hailo0`） |
| 宿主机/运行时 | HailoRT 4.23.x |
| Python | 3.11，aarch64 |
| 输入 | 416x416x3 RGB（normalize_in_net std=255，÷255） |
| 输出 | 2 个 raw head：13x13x255 (stride 32) + 26x26x255 (stride 16) |
| 类别 | 80（COCO，0-indexed） |
| 参数量 | 8.85M |
| 运算量 | 5.58G |
| HEF | Model Zoo v2.19.0，Hailo-8 |

## 构建

```bash
sudo docker build -f docker/hailo8/tiny_yolov3.dockerfile \
  -t tiny_yolov3:latest \
  src/rpi5_hailo8_tiny_yolov3
```

## 运行演示视频

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  tiny_yolov3:latest \
  python web_detection.py --model_path model/tiny_yolov3.hef --video_path video/test.mp4
```

浏览器打开 `http://<PI_IP>:8000`。使用 USB 摄像头时，挂载 `/dev/video0`，
把 `--video_path video/test.mp4` 替换为 `--camera_id 0`。

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/tiny_yolov3/predict" \
  -F "file=@test.jpg"
```

| 接口 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 健康检查 |
| `/api/models/tiny_yolov3/predict` | POST | 检测结果（JSON） |
| `/api/video_feed` | GET | MJPEG 预览流 |

## 实现说明

- **无片上 NMS**——和 SSD/efficientdet/nanodet 不同，tiny_yolov3 输出的是 raw
  YOLOv3 head。应用在 CPU 上做完整解码：sigmoid(obj+cls)、anchor 网格解码、NMS。
- 解码公式（官方 `yolo.py` `_yolo3_decode`）：
  `center = (sigmoid(raw_xy) + grid_offset) * stride`，
  `scale = exp(raw_wh) * anchor`，
  `score = sigmoid(obj) * sigmoid(cls)`（YOLOv3 obj×cls）。
- `normalize_in_net`（std=255，÷255）+ `padding_color=114`（灰色，YOLO 惯例）。
  应用在 letterbox 后直接喂原始 uint8 RGB 像素，不做手动归一化。
- 每尺度 3 个 anchor：stride 32 → [[81,82],[135,169],[344,319]]，
  stride 16 → [[23,27],[37,58],[81,82]]（来自 Model Zoo network YAML）。
- 首次推理打印各 head 的 shape，便于在硬件上核对（SOP §10）。

## 模型来源

Hailo-8 HEF 来自
[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst)
（模型 `tiny_yolov3`）。

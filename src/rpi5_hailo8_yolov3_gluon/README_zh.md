# YOLOv3-Gluon（Raspberry Pi 5 / CM5 + Hailo-8）

本模块在 Hailo-8 上运行 YOLOv3-Gluon COCO 80 类目标检测，**YOLOv3 解码在 CPU 侧完成**
（片上不做 NMS）。HEF 输出三个原始 YOLOv3 检测头，应用在 CPU 上完成网格解码
（sigmoid + anchor + NMS）。FastAPI 服务支持图片、视频文件、USB 摄像头、MJPEG
预览和 REST 预测接口。

## 兼容性

| 项目 | 版本 |
|---|---|
| 加速器 | Hailo-8 PCIe（`/dev/hailo0`） |
| HailoRT 主机/运行时 | 4.23.x |
| Python | 3.11，aarch64 |
| 输入 | 608x608x3 RGB（网络内归一化；填充色 114） |
| 输出 | 3 个原始检测头：19x19 / 38x38 / 76x76，每个 255 通道 |
| 类别 | 80 类 COCO，0 起始 |
| 参数量 | 68.79M |
| 运算量 | 140.7G |
| mAP（浮点 / Hailo-8） | 37.3 / 35.8 |
| FPS（Hailo-8，batch 1） | 38.1 |
| HEF | Model Zoo v2.20.0，Hailo-8（50,802,846 字节） |

## 构建

在仓库根目录执行：

```bash
sudo docker build -f docker/hailo8/yolov3_gluon.dockerfile \
  -t yolov3_gluon:latest \
  src/rpi5_hailo8_yolov3_gluon
```

## 运行演示视频

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  yolov3_gluon:latest \
  python web_detection.py --model_path model/yolov3_gluon.hef --video_path video/test.mp4
```

打开 `http://<开发板IP>:8000`。USB 摄像头需挂载 `/dev/video0`，并把
`--video_path video/test.mp4` 改为 `--camera_id 0`。

## REST API

```bash
curl -X POST "http://<开发板IP>:8000/api/models/yolov3_gluon/predict" \
  -F "file=@test.jpg"
```

| 端点 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 健康检查 |
| `/api/models/yolov3_gluon/predict` | POST | 检测结果（JSON） |
| `/api/video_feed` | GET | MJPEG 预览流 |

## 实现说明

- **片上不做 NMS**：HEF 输出三个原始 YOLOv3 检测头，sigmoid、anchor 网格解码和
  NMS 全部在 CPU 完成。
- 解码公式来自官方 `yolo.py` 的 `_yolo3_decode`：
  `center = (sigmoid(raw_xy) + grid_offset) * stride`，`scale = exp(raw_wh) *
  anchor`，`score = sigmoid(obj) * sigmoid(cls)`。
- 先 letterbox（填充色 114），再 BGR→RGB。模型归一化已编译进 HEF，输入原始
  uint8 RGB 即可。
- 三个 anchor 尺度来自网络 YAML：stride 32 → [[116,90],[156,198],[373,326]]，
  stride 16 → [[30,61],[62,45],[59,119]]，stride 8 → [[10,13],[16,30],[33,23]]。
- 首次推理会打印每个检测头的 shape，便于实机核对（SOP §10）。

## 模型来源

Hailo-8 HEF 来自 [Hailo Model Zoo v2.20.0](https://github.com/hailo-ai/hailo_model_zoo)
（模型 `yolov3_gluon`）。

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/yolov3_gluon.hef
```

# DAMO-YOLO（tinynasL35_M）：Raspberry Pi 5 / CM5 + Hailo-8 目标检测

本模块运行 DAMO-YOLO 目标检测（TinyNAS-L35 主干，Medium 尺寸，COCO 80 类），
使用 `nanodet_split` 后处理头（DFL 框回归）。它是 damoyolo 家族中精度最高的
Medium（L35_M）变体，与 L20_T / L25_S 输入输出契约相同，主干最深。FastAPI 服务
支持图片、视频文件、USB 摄像头、MJPEG 预览和 REST 推理接口。

## 兼容环境

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-8 PCIe（`/dev/hailo0`） |
| 宿主机/运行时 | HailoRT 4.23.x |
| Python | 3.11，aarch64 |
| 输入 | 640x640x3 RGB（归一化为 no-op，直接喂原始 uint8） |
| 输出 | 6 个头：box (80x80x68, 40x40x68, 20x20x68) + cls (80x80x81, 40x40x81, 20x20x81) |
| 类别 | 80（COCO） |
| 参数量 | 33.98M |
| 运算量 | 61.64G |
| HEF | Model Zoo v2.19.0，Hailo-8 |

宿主机驱动、固件、`libhailort.so` 和容器内 Python wheel 必须使用相同的
HailoRT 主版本和次版本。

## 构建

在仓库根目录执行：

```bash
sudo docker build -f docker/hailo8/damoyolo_tinynas_l35_m.dockerfile \
  -t damoyolo_tinynas_l35_m:latest \
  src/rpi5_hailo8_damoyolo_tinynas_l35_m
```

## 运行演示视频

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  damoyolo_tinynas_l35_m:latest \
  python web_detection.py --model_path model/damoyolo_tinynas_l35_m.hef --video_path video/test.mp4
```

浏览器打开 `http://<PI_IP>:8000`。使用 USB 摄像头时，挂载 `/dev/video0`，
并把 `--video_path video/test.mp4` 替换为 `--camera_id 0`。

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/damoyolo_tinynas_l35_m/predict" \
  -F "file=@test.jpg"
```

| 接口 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 健康检查 |
| `/api/models/damoyolo_tinynas_l35_m/predict` | POST | 检测结果（JSON） |
| `/api/video_feed` | GET | MJPEG 预览流 |

## 实现说明

- 输入输出契约和后处理与 L20_T / L25_S 完全一致，仅主干不同（TinyNAS-L35，
  Medium 尺寸，精度最高但最重）。
- HEF 使用 `normalize_in_net`（mean=0/std=1，实质 no-op）+ `padding_color=0`；
  应用在 letterbox 到 640x640 后转 BGR→RGB，直接喂原始 uint8 像素，不做手动归一化。
- 片上只做分类 sigmoid（`device_pre_post_layers.sigmoid`）。网格解码、DFL 框还原
  （regression_length=16 → 4×17=68 通道）和按类 NMS 全在 CPU 上完成，忠实复刻
  Model Zoo 官方 `nanodet.py`（split_decode + _box_decoding）。
- 首次推理会打印所有输出 vstream 的名称/shape 以及解析出的各尺度 box/cls 分组，
  便于在硬件上核对 head 分配是否正确。
- 默认置信度 0.25、IOU 0.45（预览更干净）。Model Zoo 评估用 0.05 / 0.7——
  调低滑块可查看更低置信度的框。

## 模型来源

Hailo-8 HEF 来自
[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst)
（模型 `damoyolo_tinynasL35_M`）。

# YOLOX-Tiny：Raspberry Pi 5 / CM5 + Hailo-8 目标检测

本模块运行 YOLOX-Tiny 目标检测（COCO 80 类，旷视 Megvii），使用**片上 NMS**
（Hailo HPP，`meta_arch=yolox`）。HEF 在片上完成 NMS，直接输出已解码的检测结果。
FastAPI 服务支持图片、视频文件、USB 摄像头、MJPEG 预览和 REST 推理接口。

## 兼容环境

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-8 PCIe（`/dev/hailo0`） |
| 宿主机/运行时 | HailoRT 4.23.x |
| Python | 3.11，aarch64 |
| 输入 | 416x416x3 RGB（normalize_in_net ImageNet RGB 均值/方差） |
| 输出 | 片上 NMS 张量，后 NMS shape 80x5x100 |
| 类别 | 80（COCO，0-indexed） |
| 参数量 | 5.05M |
| 运算量 | 6.44G |
| HEF | Model Zoo v2.19.0，Hailo-8 |

## 构建

```bash
sudo docker build -f docker/hailo8/yolox_tiny.dockerfile \
  -t yolox_tiny:latest \
  src/rpi5_hailo8_yolox_tiny
```

## 运行演示视频

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  yolox_tiny:latest \
  python web_detection.py --model_path model/yolox_tiny.hef --video_path video/test.mp4
```

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/yolox_tiny/predict" \
  -F "file=@test.jpg"
```

## 实现说明

- 片上 NMS（HPP，`meta_arch=yolox`）；应用只解析后 NMS 张量。
- `normalize_in_net`（ImageNet RGB 均值/方差），无 input_conversion → 喂原始 uint8 RGB。
- `padding_color=114`（灰色，YOLOX 惯例）。
- HailoRT 参差 NMS-by-score 输出由解析器处理。
- 类别映射：`cls_id`(0..79) → 标准 COCO 80 类列表直接索引。

## 模型来源

[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst)
（模型 `yolox_tiny`，源：[Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)）。

# YOLO11n-seg - 实例分割

YOLO11n-seg（2.9M 参数）在 Hailo-8 上运行。

## 模型

| 属性 | 值 |
|------|-----|
| 架构 | YOLO11n-seg（anchor-free，one2many 头，stride 8/16/32） |
| 输入 | 640x640x3 RGB（letterbox 填充 114） |
| HEF 输出 | 10 个原始张量：每个 stride 的 64 通道 DFL 框、80 通道类别分数、32 通道 mask 系数，另加 160x160x32 mask 原型 |
| 参数量 | 2.9M |
| 运算量 | 9.81G |
| mAP-seg（浮点 / Hailo-8） | 32.1 / 31.1 |
| FPS（batch 1 / 8） | 132 / 287 |
| HEF | Hailo Model Zoo v2.20.0，Hailo-8（7,915,205 字节） |
| 许可证 | AGPL-3.0（Ultralytics） |

## 快速开始

运行基线：Python 3.11 与 HailoRT 4.23.0。在仓库根目录执行构建命令。

```bash
docker build -t yolov11n_seg -f docker/hailo8/yolov11n_seg.dockerfile src/rpi5_hailo8_yolov11n_seg

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  yolov11n_seg
```

## API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 预览 |
| `/api/video_feed` | GET | MJPEG 流 |
| `/api/models/yolov11n_seg/predict` | POST | 框级检测结果（JSON） |

HEF 输出的是原始头，没有片上 NMS：主机侧执行 DFL 框解码、sigmoid + 按类 NMS，
以及 mask 组装（系数乘原型，再按框裁剪）。实例 mask 在 MJPEG 预览中绘制；REST
返回框、类别和置信度。

## 来源

HEF 来自 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) v2.20.0（Hailo-8）：

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/yolov11n_seg.hef
```

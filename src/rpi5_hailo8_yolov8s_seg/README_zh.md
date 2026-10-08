# YOLOv8s-seg - 实例分割

YOLOv8s-seg（11.8M 参数），Hailo-8 平台。

## 模型

| 属性 | 值 |
|------|-----|
| 架构 | YOLOv8s-seg（anchor-free，one2many head，stride 8/16/32） |
| 输入 | 640x640x3 RGB（letterbox 填充 114） |
| HEF 输出 | 10 个原始张量：每个 stride 的 64 通道 DFL 框回归、80 通道类别分数、32 通道掩码系数，另有 160x160x32 掩码原型 |
| 参数量 | 11.8M |
| 运算量 | 42.6G |
| 参考 mAP | 36.634（COCO 实例分割，全精度，Hailo Model Zoo） |
| 许可 | AGPL-3.0（Ultralytics） |
| 格式 | HEF（Hailo-8） |

## 快速开始

运行基线：Python 3.11 与 HailoRT 4.23.0。构建命令在仓库根目录执行。

```bash
docker build -t yolov8s_seg -f docker/hailo8/yolov8s_seg.dockerfile src/rpi5_hailo8_yolov8s_seg

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  yolov8s_seg
```

## API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 预览 |
| `/api/video_feed` | GET | MJPEG 流 |
| `/api/models/yolov8s_seg/predict` | POST | 框级检测结果（JSON） |

HEF 输出原始分支头、无片上 NMS（与 yolov8 检测编译不同）：主机侧完成 DFL 框解码、
sigmoid 与逐类 NMS、掩码合成（系数 x 原型，按框裁剪）。实例掩码在 MJPEG 预览中绘制；
REST 返回框、类别与置信度。

## 来源

HEF 来自 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) v2.19.0（Hailo-8）：

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/yolov8s_seg.hef
```

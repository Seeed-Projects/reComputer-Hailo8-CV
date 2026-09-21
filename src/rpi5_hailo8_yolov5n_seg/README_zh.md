# YOLOv5n-seg - 实例分割

YOLOv5n-seg（anchor-based 实例分割，1.99M 参数），Hailo-8 平台。

## 模型信息

| 属性 | 值 |
|------|-----|
| 架构 | YOLOv5n-seg（3 anchors，stride 8/16/32） |
| 输入 | 640×640×3 RGB |
| 输出 | 80 类检测框 + 实例掩码（proto 160x160x32） |
| 参数量 | 1.99M |
| 硬件 mAP | 22.9（COCO，Hailo Model Zoo 参考值） |
| 格式 | HEF (Hailo-8) |

## 快速开始

运行时基线：Python 3.11、HailoRT 4.23.0。请在仓库根目录执行构建命令。

```bash
docker build -t yolov5n_seg -f docker/hailo8/yolov5n_seg.dockerfile src/rpi5_hailo8_yolov5n_seg

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  yolov5n_seg
```

## API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 预览 |
| `/api/video_feed` | GET | MJPEG 流 |
| `/api/models/yolov5n_seg/predict` | POST | 检测框 + 实例掩码（JSON） |

## 来源

HEF 来自 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)。

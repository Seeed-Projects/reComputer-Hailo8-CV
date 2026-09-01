# YOLO26x-seg - 实例分割

YOLO26x-seg（57.7M 参数），Hailo-8 平台。

## 模型信息

| 属性 | 值 |
|------|-----|
| 架构 | YOLO26x-seg |
| 输入 | 640×640×3 RGB |
| HEF 输出 | 边界框 + 实例掩码张量 (COCO 80类) |
| 参数量 | 57.7M |
| 格式 | HEF (Hailo-8) |

## 快速开始

运行时基线：Python 3.11、HailoRT 4.23.0。请在仓库根目录执行构建命令。

```bash
docker build -t yolo26x-seg -f docker/hailo8/yolo26x_seg.dockerfile src/hailo8_yolo26x_seg

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  yolo26x-seg
```

## API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 预览 |
| `/api/video_feed` | GET | MJPEG 视频流 |
| `/api/models/yolo26x_seg/predict` | POST | 框级检测结果 (JSON) |

当前 Web 后处理仅输出框级检测结果；实例掩码解码仍需在目标硬件上验证。仓库内的 YOLO26m HEF 相对所述模型规模异常偏小，发布前需确认工件。

## 来源

HEF 模型来自 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)。

# LightFace Slim - 人脸检测

超轻量人脸检测（0.26M 参数）在 Hailo-8 上的部署。

## 模型

| 属性 | 值 |
|------|-----|
| 架构 | Ultra-Light-Fast-Generic-Face-Detector-1MB（SSD 结构，stride 8/16/32/64） |
| 输入 | 240x320x3 uint8 BGR（等比缩放 + 右下补零；mean 127 / std 128 已编译进 HEF） |
| 输出 | 8 个原始分支头：4 个尺度 × {框增量、背景+人脸 logits}；共 4,420 个锚点 |
| 解码 | 主机侧 SSD 解码（方差 10/5）+ softmax + 贪心 NMS（`meta_arch=retinaface`） |
| 参数量 | 0.26M |
| 运算量 | 0.16G |
| 参考 mAP | 39.71（WIDER FACE，全精度，Hailo Model Zoo） |
| 许可 | MIT（Ultra-Light-Fast-Generic-Face-Detector-1MB） |
| 格式 | HEF（Hailo-8） |

## 快速开始

运行基线：Python 3.11 与 HailoRT 4.23.0。构建命令在仓库根目录执行。

```bash
docker build -t lightface_slim -f docker/hailo8/lightface_slim.dockerfile src/rpi5_hailo8_lightface_slim

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  lightface_slim
```

## API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 预览 |
| `/api/video_feed` | GET | MJPEG 流 |
| `/api/models/lightface_slim/predict` | POST | 人脸框 + 置信度（JSON） |

HEF 输出原始分支头、无片上 NMS：主机侧完成 SSD 解码（方差 10/5）、背景/人脸
logits 的 softmax、阈值筛选与贪心 NMS。首次推理会打印全部输出张量的名称、形状与
数值范围，便于在容器日志中确认实际布局。

## 来源

HEF 来自 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) v2.19.0（Hailo-8）：

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/lightface_slim.hef
```

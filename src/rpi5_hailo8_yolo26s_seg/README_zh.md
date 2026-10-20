# YOLO26s-seg - 实例分割

YOLO26s-seg（10.4M 参数），Hailo-8 平台。模块会解码原始分割头、组装实例掩码
（掩码系数 × prototype，按框裁剪）并绘制到带标注的预览里。

## 模型信息

| 属性 | 值 |
|------|-----|
| 架构 | YOLO26s-seg |
| 输入 | 640x640x3 RGB |
| HEF 输出 | 10 个原始张量：每个 stride 的 4 通道框（l,t / r,b 距离）、80 通道类别 logits、32 通道掩码系数，外加 160x160x32 prototype |
| 参数量 | 10.4M |
| 运算量 | 37G |
| 格式 | HEF (Hailo-8) |
| HEF 大小 | 23,424,755 字节 |
| HEF SHA256 | `aa49b1c247a4d75ad221c51d43c3f740af1715b026717f871a50976e5a37be66` |

## 快速开始

运行基线：Python 3.11 与 HailoRT 4.23.0。在仓库根目录执行构建命令。

```bash
docker build -t yolo26s-seg -f docker/hailo8/yolo26s_seg.dockerfile src/rpi5_hailo8_yolo26s_seg

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo8-cv/yolo26s_seg:latest
```

## API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 预览 |
| `/api/video_feed` | GET | MJPEG 视频流 |
| `/api/models/yolo26s_seg/predict` | POST | 检测结果（JSON：类别、置信度、框） |

HEF 暴露原始头（没有片上 NMS）：宿主做 sigmoid、one2one 头的两段式 top-k
（post_nms_topk=100，无 NMS）、regression_length=1 的框解码以及掩码组装。
JSON 只携带框；实例掩码会合成到带标注的预览和离线视频输出中。

## 来源

HEF 来自 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) v2.19.0 (Hailo-8)：

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/yolo26s_seg.hef
```

Hailo-8 的 zoo 编译了 yolo26s_seg / yolo26m_seg / yolo26l_seg / yolo26x_seg（不存在 n_seg 构建）。

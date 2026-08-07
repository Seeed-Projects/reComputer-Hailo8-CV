# CenterNet（resnet_v1_18）：Raspberry Pi 5 / CM5 + Hailo-8 目标检测

本模块运行 CenterNet 目标检测（ResNet-18 主干，COCO 80 类），使用片上
`max_finder` 后处理变体。FastAPI 服务支持图片、视频文件、USB 摄像头、
MJPEG 预览和 REST 推理接口。

## 兼容环境

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-8 PCIe（`/dev/hailo0`） |
| 宿主机/运行时 | HailoRT 4.23.x |
| Python | 3.11，aarch64 |
| 输入 | 512x512x3 RGB（归一化已编译进 HEF） |
| 输出 | 3 个头：wh (128x128x2)、reg (128x128x2)、稀疏热图 (128x128x80) |
| 类别 | 80（COCO） |
| 参数量 | 14.22M |
| HEF | Model Zoo v2.19.0，Hailo-8 |

宿主机驱动、固件、`libhailort.so` 和容器内 Python wheel 必须使用相同的
HailoRT 主版本和次版本。

## 构建

在仓库根目录执行：

```bash
sudo docker build -f docker/hailo8/centernet_resnet_v1_18_postprocess.dockerfile \
  -t centernet_resnet_v1_18_postprocess:latest \
  src/rpi5_hailo8_centernet_resnet_v1_18_postprocess
```

## 运行演示视频

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  centernet_resnet_v1_18_postprocess:latest \
  python web_detection.py --model_path model/centernet_resnet_v1_18_postprocess.hef --video_path video/test.mp4
```

浏览器打开 `http://<PI_IP>:8000`。使用 USB 摄像头时，挂载 `/dev/video0`，
并把 `--video_path video/test.mp4` 替换为 `--camera_id 0`。

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/centernet_resnet_v1_18_postprocess/predict" \
  -F "file=@test.jpg"
```

| 接口 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 健康检查 |
| `/api/models/centernet_resnet_v1_18_postprocess/predict` | POST | 检测结果（JSON） |
| `/api/video_feed` | GET | MJPEG 预览流 |

## 实现说明

- HEF 使用 `normalize_in_net=true`（ImageNet 均值/方差，RGB），应用在 letterbox
  后直接传入原始 uint8 RGB 像素，不做手动归一化。
- 热图头已经是稀疏的：片上 `max_finder` + 置信度阈值（0.2）只保留高于阈值的
  局部极大值。解码器取这些峰值，在每个峰值处读取 `wh`/`reg`，按 Hailo Model
  Zoo 官方公式重建框（步长 4，128x128 特征图 -> 512x512 输入）。
- 首次推理会打印所有输出 vstream 的名称/shape 以及解析出的 hm/wh/reg 映射，
  便于在硬件上核对 head 分配是否正确。

## 模型来源

Hailo-8 HEF 来自
[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst)。

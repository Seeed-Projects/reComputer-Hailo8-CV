# ViTPose-Small：Raspberry Pi 5 / CM5 + Hailo-8 单人姿态估计

本模块运行 ViTPose-Small 单人 2D 姿态估计（17 个 COCO 关键点）。HEF 输出
单张热图（64×48×17），应用对每通道做 argmax 提取关键点坐标，缩放到输入空间，
绘制 COCO 骨架。FastAPI 服务支持图片、视频文件、USB 摄像头、MJPEG 预览和
REST 推理接口。

## 兼容环境

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-8 PCIe（`/dev/hailo0`） |
| 宿主机/运行时 | HailoRT 4.23.x |
| Python | 3.11，aarch64 |
| 输入 | 256x192x3 RGB（normalize_in_net ImageNet RGB 均值/方差） |
| 输出 | 热图 64x48x17（17 个 COCO 关键点） |
| 参数量 | 24.29M |
| 运算量 | 17.17G |
| HEF | Model Zoo v2.19.0，Hailo-8 |

## 构建

```bash
sudo docker build -f docker/hailo8/vit_pose_small.dockerfile \
  -t vit_pose_small:latest \
  src/rpi5_hailo8_vit_pose_small
```

## 运行演示视频

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  vit_pose_small:latest \
  python web_detection.py --model_path model/vit_pose_small.hef --video_path video/test.mp4
```

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/vit_pose_small/predict" \
  -F "file=@test.jpg"
```

## 实现说明

- **单人姿态**：模型假设人物居中在裁剪框内。多人场景需先用检测器（YOLO）裁剪
  每个人，再逐人跑 ViTPose。
- **后处理**：热图每通道 argmax → 64×48 空间坐标 → 缩放到 256×192 输入 →
  反 letterbox 到原图。无 DARK 亚像素精修（演示足够）。
- `normalize_in_net`（ImageNet RGB 均值/方差）+ 无 input_conversion → 喂原始
  uint8 RGB。

## 模型来源

[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst)
（模型 `vit_pose_small`）。

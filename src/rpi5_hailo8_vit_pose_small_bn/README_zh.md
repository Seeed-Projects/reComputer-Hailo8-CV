# ViTPose-Small-BN：Raspberry Pi 5 / CM5 + Hailo-8 单人姿态估计

与 ViTPose-Small 同架构，BatchNorm 变体（24.32M 参数，AP 72.01）。I/O 完全
一致：256×192 RGB 输入，64×48×17 热图输出，17 个 COCO 关键点。

## 兼容环境

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-8 PCIe（`/dev/hailo0`） |
| 宿主机/运行时 | HailoRT 4.23.x |
| Python | 3.11，aarch64 |
| 输入 | 256x192x3 RGB |
| 输出 | 热图 64x48x17 |
| 参数量 | 24.32M |
| HEF | Model Zoo v2.19.0，Hailo-8 |

## 构建

```bash
sudo docker build -f docker/hailo8/vit_pose_small_bn.dockerfile \
  -t vit_pose_small_bn:latest \
  src/rpi5_hailo8_vit_pose_small_bn
```

## 运行

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  vit_pose_small_bn:latest \
  python web_detection.py --model_path model/vit_pose_small_bn.hef --video_path video/test.mp4
```

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/vit_pose_small_bn/predict" \
  -F "file=@test.jpg"
```

## 实现说明

- 与 vit_pose_small 完全相同的 I/O 和后处理；仅 BN 变体（AP 略低：72.01 vs 74.16）。
- 单人姿态：热图每通道 argmax → 关键点坐标 → 缩放 → 反 letterbox → 画骨架。

## 模型来源

[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst)
（模型 `vit_pose_small_bn`）。

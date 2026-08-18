# Depth-Anything-ViTS：Raspberry Pi 5 / CM5 + Hailo-8 深度估计

与 Depth-Anything-V2-ViTS 同架构（V1 版本），AbsRel 0.13（比 V2 的 0.147 略好）。
I/O 完全一致：224×224 RGB → 224×224×1 相对深度图，INFERNO 着色，深度统计 API。

## 兼容环境

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-8 PCIe（`/dev/hailo0`） |
| 宿主机/运行时 | HailoRT 4.23.x |
| Python | 3.11，aarch64 |
| 输入 | 224x224x3 RGB |
| 输出 | 深度图 224x224x1（相对深度） |
| 参数量 | 24.2M |
| AbsRel | 0.13 |
| HEF | Model Zoo v2.19.0，Hailo-8 |

## 构建

```bash
sudo docker build -f docker/hailo8/depth_anything_vits.dockerfile \
  -t depth_anything_vits:latest \
  src/rpi5_hailo8_depth_anything_vits
```

## 运行

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  depth_anything_vits:latest \
  python web_detection.py --model_path model/depth_anything_vits.hef --video_path video/test.mp4
```

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/depth_anything_vits/predict" \
  -F "file=@test.jpg"
```

## 实现说明

- 与 V2-ViTS 完全相同的 I/O 和后处理；V1 精度略好（AbsRel 0.13 vs 0.147）。
- 零样本相对深度，INFERNO 着色，alpha 混合叠加。
- `normalize_in_net` ImageNet RGB，喂原始 uint8 RGB。

## 模型来源

[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst)
（模型 `depth_anything_vits`）。

# Depth-Anything-V2-ViTS：Raspberry Pi 5 / CM5 + Hailo-8 深度估计

本模块运行 Depth-Anything-V2-Small（ViT-S 主干）进行零样本单目深度估计。
HEF 输出 224×224×1 相对深度图，应用进行归一化、INFERNO 着色、alpha 混合到
原图。FastAPI 服务支持图片、视频文件、USB 摄像头、MJPEG 预览和 REST 推理。

## 兼容环境

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-8 PCIe（`/dev/hailo0`） |
| 宿主机/运行时 | HailoRT 4.23.x |
| Python | 3.11，aarch64 |
| 输入 | 224x224x3 RGB |
| 输出 | 深度图 224x224x1（相对深度，零样本） |
| 参数量 | 24.2M |
| 运算量 | 16.7G |
| AbsRel | 0.147 |
| HEF | Model Zoo v2.19.0，Hailo-8 |

## 构建

```bash
sudo docker build -f docker/hailo8/depth_anything_v2_vits.dockerfile \
  -t depth_anything_v2_vits:latest \
  src/rpi5_hailo8_depth_anything_v2_vits
```

## 运行

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  depth_anything_v2_vits:latest \
  python web_detection.py --model_path model/depth_anything_v2_vits.hef --video_path video/test.mp4
```

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/depth_anything_v2_vits/predict" \
  -F "file=@test.jpg"
```

响应返回深度统计值（min、max、mean、std）。

## 实现说明

- **零样本相对深度**：输出是相对深度（非公制），值越大越远。应用按帧
  min-max 归一化用于可视化。
- **可视化**：INFERNO 着色（暗=近，亮=远），alpha 混合到原图。滑块控制混合比例。
- `normalize_in_net`（ImageNet RGB 均值/方差），无 input_conversion → 喂原始 uint8 RGB。

## 模型来源

[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst)
（模型 `depth_anything_v2_vits`，源：[Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)）。

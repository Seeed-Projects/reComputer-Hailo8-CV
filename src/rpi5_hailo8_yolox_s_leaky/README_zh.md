# YOLOX-S-Leaky：Raspberry Pi 5 / CM5 + Hailo-8 目标检测

与 YOLOX-Tiny 同架构，S 主干（640×640 输入，精度更高）。片上 NMS（Hailo HPP，
`meta_arch=yolox`），COCO 80 类。

## 兼容环境

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-8 PCIe（`/dev/hailo0`） |
| 宿主机/运行时 | HailoRT 4.23.x |
| Python | 3.11，aarch64 |
| 输入 | 640x640x3 RGB（normalize_in_net ImageNet RGB） |
| 输出 | 片上 NMS 张量，后 NMS shape 80x5x100 |
| 类别 | 80（COCO，0-indexed） |
| HEF | Model Zoo v2.19.0，Hailo-8 |

## 构建

```bash
sudo docker build -f docker/hailo8/yolox_s_leaky.dockerfile \
  -t yolox_s_leaky:latest \
  src/rpi5_hailo8_yolox_s_leaky
```

## 运行

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  yolox_s_leaky:latest \
  python web_detection.py --model_path model/yolox_s_leaky.hef --video_path video/test.mp4
```

## 模型来源

[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst)
（模型 `yolox_s_leaky`，源：[Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)）。
# YOLOX-L-Leaky：Raspberry Pi 5 / CM5 + Hailo-8 目标检测

与 YOLOX-Tiny/S-Leaky 同架构，L 主干（640×640 输入，54.17M 参数，精度最高）。
片上 NMS（Hailo HPP，`meta_arch=yolox`），COCO 80 类。

## 运行

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  yolox_l_leaky:latest \
  python web_detection.py --model_path model/yolox_l_leaky.hef --video_path video/test.mp4
```

## 模型来源

[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst)
（模型 `yolox_l_leaky`，源：[Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)）。
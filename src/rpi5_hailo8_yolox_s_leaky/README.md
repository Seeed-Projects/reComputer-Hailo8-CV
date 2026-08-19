# YOLOX-S-Leaky on Raspberry Pi 5 / CM5 + Hailo-8

Same architecture as YOLOX-Tiny but with a larger S backbone (640x640 input,
higher accuracy). On-chip NMS (Hailo HPP, `meta_arch=yolox`), COCO 80 classes.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-8 PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 4.23.x |
| Python | 3.11, aarch64 |
| Input | 640x640x3 RGB (normalize_in_net ImageNet RGB) |
| Output | on-chip NMS tensor, post-NMS shape 80x5x100 |
| Classes | 80 (COCO, 0-indexed) |
| HEF | Model Zoo v2.19.0, Hailo-8 |

## Build

```bash
sudo docker build -f docker/hailo8/yolox_s_leaky.dockerfile \
  -t yolox_s_leaky:latest \
  src/rpi5_hailo8_yolox_s_leaky
```

## Run

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  yolox_s_leaky:latest \
  python web_detection.py --model_path model/yolox_s_leaky.hef --video_path video/test.mp4
```

## Model source

[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst)
(model `yolox_s_leaky`, source: [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)).
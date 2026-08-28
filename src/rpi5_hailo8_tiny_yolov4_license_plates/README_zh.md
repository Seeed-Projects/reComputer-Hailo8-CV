# CM5 + Hailo-8 Tiny-YOLOv4 车牌检测

本模块使用 Hailo Model Zoo v2.17 官方 Hailo-8 HEF，提供内置视频、USB 摄像头、图片/视频 REST 推理、MJPEG 预览和离线视频分析。

## 模型信息

- 输入：416x416x3 RGB uint8，归一化已编译进 HEF。
- 输出：13x13x18 与 26x26x18 两个 raw Tiny-YOLOv4 head。
- 类别：`license_plate`。
- 后处理：sigmoid、anchor/grid 解码、阈值过滤和 CPU NMS。
- Model Zoo 指标：74.083 mAP（Hailo 内部车牌数据集）。

## 展示视频

`video/test.mp4` 是 Hailo TAPPAS v3.29 官方 LPR 默认视频 `lpr_ayalon.mp4` 的 1280x720 H.264 转码版本：

https://hailo-tappas.s3.eu-west-2.amazonaws.com/v3.29/general/media/lpr_ayalon.mp4

## 构建运行

```bash
sudo docker build -f docker/hailo8/tiny_yolov4_license_plates.dockerfile \
  -t tiny_yolov4_license_plates:latest \
  src/rpi5_hailo8_tiny_yolov4_license_plates

sudo docker run --rm --name cm5-hailo8-lp-detector --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  tiny_yolov4_license_plates:latest
```

访问 `http://<Board_IP>:8000`。图片接口为 `/api/models/tiny_yolov4_license_plates/predict`。

## 验证状态

本地已做 Python 语法和静态元数据检查；CM5 + Hailo-8 真实推理、输出 tensor、Docker、摄像头和坐标对齐仍需实机验收。

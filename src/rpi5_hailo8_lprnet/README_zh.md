# CM5 + Hailo-8 LPRNet 车牌识别流水线

本模块采用两级流水线：Tiny-YOLOv4 从整帧检测车牌，再把原始分辨率车牌区域交给 LPRNet 识别数字。两个 HEF 通过 HailoRT round-robin 调度和共享 VDevice 组使用同一块 Hailo-8。

## 模型契约

- 检测：416x416x3 RGB，两个 raw YOLO head，CPU 完成 sigmoid、anchor 解码和 NMS。
- 识别：300x75x3 BGR 车牌裁剪，输出 5x19x11；沿第 0 维求均值，再做 softmax 和 CTC greedy decode。
- 字符表：`0123456789-`。平均字符置信度不低于 0.90 且 CTC 去重后不少于 7 位才接受。
- OCR 前使用 TAPPAS 官方的拉普拉斯方差门限 `quality >= 100`。

官方 LPRNet **不支持字母和中国省份字符**，因此不能用于“省份+字母+5 位”的中国蓝牌完整识别。

## 展示视频

`video/test.mp4` 是 Hailo TAPPAS v3.29 官方默认 LPR 视频 `lpr_ayalon.mp4` 的 1280x720 H.264 转码版本：

https://hailo-tappas.s3.eu-west-2.amazonaws.com/v3.29/general/media/lpr_ayalon.mp4

## 构建运行

```bash
sudo docker build -f docker/hailo8/lprnet.dockerfile \
  -t lprnet:latest src/rpi5_hailo8_lprnet

sudo docker run --rm --name cm5-hailo8-lprnet --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  lprnet:latest
```

访问 `http://<Board_IP>:8000`，图片接口为 `/api/models/lprnet/predict`。

## 验证状态

本地已完成 Python 语法和素材检查；双 HEF 调度、真实 tensor、OCR 结果、Docker 和端到端效果仍需 CM5 + Hailo-8 实机验收。

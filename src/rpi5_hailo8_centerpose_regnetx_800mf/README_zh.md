# CenterPose RegNetX-800MF：Raspberry Pi 5 + Hailo-8

本模块在 Hailo-8 上运行 CenterPose RegNetX-800MF，实现多人姿态估计。视频、
摄像头、MJPEG 和 REST 服务复用 `reComputer-R20-CV` 的 `yolov8_pose` 模板，
推理前处理与后处理已替换为 CenterPose 专用实现。

## 兼容版本

| 项目 | 版本 |
|---|---|
| 加速器 | Hailo-8 PCIe（`/dev/hailo0`） |
| 宿主机 HailoRT | 4.23.x |
| Python | 3.11，aarch64 |
| 输入 | 512×512×3 BGR |
| 输出 | 人体框与 17 个 COCO 关键点 |
| 参数量 | 12.31M |
| HEF | Model Zoo v2.19.0，Hailo-8 |

宿主机驱动、固件、`libhailort.so` 与容器内 Python wheel 的 HailoRT
主/次版本必须一致。

## 构建

在仓库根目录运行：

```bash
sudo docker build -f docker/hailo8/centerpose_regnetx_800mf.dockerfile \
  -t centerpose_regnetx_800mf:latest \
  src/rpi5_hailo8_centerpose_regnetx_800mf
```

## 运行演示视频

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  centerpose_regnetx_800mf:latest \
  python web_detection.py \
    --model_path model/centerpose_regnetx_800mf.hef \
    --video_path video/test.mp4
```

浏览器打开 `http://<树莓派IP>:8000`。USB 摄像头模式需额外挂载
`/dev/video0`，并把视频参数替换为 `--camera_id 0`。

## REST API

```bash
curl -X POST "http://<树莓派IP>:8000/api/models/centerpose_regnetx_800mf/predict" \
  -F "file=@test.jpg"
```

## 实现说明

- `web_detection.py` 将 v2.19 HEF 的 `conv60`–`conv65` 六个输出映射为
  `hm`、`wh`、`hps`、`reg`、`hm_hp` 和 `hp_offset`。
- HEF 已内置原始 BGR 均值归一化，因此 OpenCV 图像不转换为 RGB。
- 第一次推理会打印全部输出名称与形状，方便在设备上核验。


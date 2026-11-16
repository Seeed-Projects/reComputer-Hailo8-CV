# ByteTrack-Nano (MOT17)（CM5 + Hailo-8）

本模块在 **Hailo-8** 上通过 HailoRT 4.23.0 运行 MOT17 风格的行人跟踪检测器。
HEF 输出九个原始 YOLOX 检测头（每个 stride 的 box、objectness、class）；主机侧完成
框解码、NMS，再送入 ByteTrack 关联，MJPEG 预览会为每个人绘制稳定的 track ID、
类别和置信度。

## 兼容性

| 项目 | 值 |
|---|---|
| 加速器 | Hailo-8 PCIe（`/dev/hailo0`） |
| HailoRT 主机/运行时 | 4.23.0 |
| Python | 3.11，aarch64 |
| 输入 | 608x1088x3 RGB（letterbox，填充 114；ImageNet 归一化在 HEF 内） |
| 输出 | 9 个原始头：19x34 / 38x68 / 76x136，每个 box(4) + obj(1) + class(1) |
| 类别 | 1 类（person，MOT17） |
| 参数量 | 0.90M |
| 运算量 | 4.12G |
| MOTA（浮点 / Hailo-8） | 70.1 / 69.0 |
| FPS（Hailo-8，batch 1） | 63.7 |
| HEF | Model Zoo v2.20.0，Hailo-8（9,834,452 字节） |

## 构建

在仓库根目录执行：

```bash
sudo docker build -f docker/hailo8/bytetrack_nano_mot17.dockerfile \
  -t bytetrack_nano_mot17:latest \
  src/rpi5_hailo8_bytetrack_nano_mot17
```

## 运行演示视频

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  bytetrack_nano_mot17:latest \
  python web_detection.py --model_path model/bytetrack_nano_mot17.hef --video_path video/test.mp4
```

打开 `http://<开发板IP>:8000` 查看带 track ID 的叠加。内置的 `video/test.mp4`
是 Hailo 官方 example.mp4 街景演示片段。USB
摄像头需挂载 `/dev/video0`，并把 `--video_path ...` 改为 `--camera_id 0`。

## REST API

```bash
curl -X POST "http://<开发板IP>:8000/api/models/bytetrack_nano_mot17/predict" \
  -F "file=@test.jpg"
```

每个预测结果包含 `track_id`、`class`、`confidence` 和 `box`。

## 实现说明

- HEF 输出九个 YOLOX 风格分支（stride 32/16/8 的 box/obj/class），片上不做 NMS。
- 主机解码：`center = (raw_xy + grid) * stride`，`size = exp(raw_wh) * stride`，
  `score = obj * class`（HEF 已完成 sigmoid）。框按 0.1 过滤，NMS 阈值 0.7。
- 框在 `py_utils/byte_tracker.py` 中做 ByteTrack 关联（Kalman 预测 + 高/低分 IoU
  匹配），只绘制已激活的轨迹。
- 预处理 letterbox 到 608x1088（填充 114），BGR→RGB；ImageNet 归一化已编译进 HEF。
- 这是 MOT17 单类 person 检测器，不是通用 80 类检测模型。

## 模型来源

Hailo-8 HEF 来自 [Hailo Model Zoo v2.20.0](https://github.com/hailo-ai/hailo_model_zoo)
（模型 `bytetrack_nano_mot17`）。

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/bytetrack_nano_mot17.hef
```

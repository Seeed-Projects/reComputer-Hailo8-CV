# YOLO26n on CM5 + Hailo-8

YOLO26n 在 **Hailo-8** 上通过 HailoRT 执行 COCO 80 类目标检测。它是 one2one 头
（每个格子一个预测、无锚框、无需 NMS），应用解码原始分支头并采用 ultralytics
的两段式 top-k 选择；若编译产物带片上 HPP NMS 也会按该布局解析。FastAPI 服务
提供图像预测、视频与摄像头输入、MJPEG 预览以及离线视频分析。

## 兼容性

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-8 PCIe（`/dev/hailo0`） |
| HailoRT 运行时 | 4.23.x |
| Python | 3.11, aarch64 |
| 输入 | 640x640x3 RGB（normalize_in_net mean 0 / std 255，填充 114） |
| 输出 | 原始分支头 **或** 片上 NMS（首次推理打印实际布局） |
| 类别 | 80（COCO，0 基） |
| 参数量 | 2.4M |
| 运算量 | 5.5G |
| HEF | Hailo Model Zoo v2.19.0 |
| 模型许可 | AGPL-3.0（上游 ultralytics/ultralytics） |

宿主机驱动、固件、`libhailort.so` 与 Python wheel 必须使用同一 HailoRT 大版本。

## 构建

在仓库根目录执行：

```bash
sudo docker build -f docker/hailo8/yolo26n.dockerfile \
  -t yolo26n:latest \
  src/rpi5_hailo8_yolo26n
```

## 运行演示视频

```bash
sudo docker run --rm \
  --name cm5-hailo8-yolo26n \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo8-cv/yolo26n:latest \
  python web_detection.py --model_path model/yolo26n.hef --video_path video/test.mp4
```

浏览器打开 `http://<BOARD_IP>:8000`。

使用 USB 摄像头时挂载 `/dev/video0`，并把 `--video_path ...` 换成 `--camera_id 0`。

## REST API

```bash
curl -X POST "http://<BOARD_IP>:8000/api/models/yolo26n/predict" \
  -F "file=@bus.jpg" -F "conf=0.25"
```

| 接口 | 方法 | 用途 |
|---|---|---|
| `/` | GET | 网页预览界面 |
| `/api/models/yolo26n/predict` | POST | 检测结果（JSON） |
| `/api/video_feed` | GET | MJPEG 预览流 |
| `/api/config` | GET / POST | 读取或修改运行时阈值 |
| `/api/video/upload` | POST | 上传源视频 |
| `/api/video/analyze` | POST | 启动离线视频分析 |
| `/api/video/status` | GET | 读取分析进度 |
| `/api/video/download/{filename}` | GET | 下载标注后的结果 |

## 实现说明

- 原始头布局：每个 stride 32/16/8 各一个 4 通道框张量（格心到 l、t / r、b 的
  距离，stride 单位，`regression_length=1`）和一个 80 通道类别张量。宿主侧做
  sigmoid、两段式 top-k（`post_nms_topk=300`，不做 NMS），并把框裁剪到输入范围。
- 片上 NMS 布局：改为解析 post-NMS 张量（逐类紧凑缓冲区、`Cx5xD` / `CxDx5`
  稠密布局、ragged NMS-by-score 列表），宿主侧不再做 NMS。
- 预处理把画面 letterbox 到 640x640x3，填充灰色（114），BGR 转 RGB 后送入原始
  uint8；`/255` 归一化已编译进 HEF。
- `cls_id`（0..79）直接索引标准 COCO 类别表。
- 首次推理会打印布局与框/分数范围（`[YOLO26] layout=..., outputs=[...]`）。

## 硬件验收清单

1. `hailortcli --version` 为 4.23.x 且 `/dev/hailo0` 存在。
2. 启动日志打印 HEF 输入尺寸，首次推理打印输出布局。
3. 演示视频能显示带类别标签的 COCO 检测框。
4. `POST /api/models/yolo26n/predict` 返回含类别、置信度和框的结果。
5. USB 摄像头模式预览持续刷新，无卡死帧。

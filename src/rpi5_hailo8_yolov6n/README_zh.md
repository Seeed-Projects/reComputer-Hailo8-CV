# YOLOv6n on CM5 + Hailo-8

YOLOv6n 在 **Hailo-8** 上通过 HailoRT 执行 COCO 80 类目标检测。编译出的 HEF
可能直接给出片上 HPP NMS 结果，也可能暴露九个原始分支头；应用对两种布局都能
处理（Model Zoo `base/yolov6.yaml`：`hpp=true`、`meta_arch=yolo_v6`、
`score_threshold=0.03`、`nms_iou_thresh=0.65`）。FastAPI 服务提供图像预测、
视频与摄像头输入、MJPEG 预览以及离线视频分析。

## 兼容性

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-8 PCIe（`/dev/hailo0`） |
| HailoRT 运行时 | 4.23.0 |
| Python | 3.11, aarch64 |
| 输入 | 640x640x3 RGB (normalize_in_net mean 0 / std 255, padding 114) |
| 输出 | 片上 NMS 张量 **或** 九个原始分支头（以首次推理日志为准） |
| 类别 | 80（COCO，0 基） |
| 参数量 | 4.32M |
| 运算量 | 11.12G |
| HEF | Hailo Model Zoo v5.4.0 (Hailo-10H) / v2.19.0 (Hailo-8) |
| 模型许可 | GPL-3.0 (upstream: meituan/YOLOv6) |

宿主机驱动、固件、`libhailort.so` 与 Python wheel 必须使用同一 HailoRT 大版本。

## 构建

在仓库根目录执行：

```bash
sudo docker build -f docker/hailo8/yolov6n.dockerfile \
  -t yolov6n:latest \
  src/rpi5_hailo8_yolov6n
```

## 运行演示视频

```bash
sudo docker run --rm \
  --name cm5-hailo8-yolov6n \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo8-cv/yolov6n:latest \
  python web_detection.py --model_path model/yolov6n.hef --video_path video/test.mp4
```

浏览器打开 `http://<BOARD_IP>:8000`。

使用 USB 摄像头时挂载 `/dev/video0`，并把 `--video_path ...` 换成 `--camera_id 0`。

## REST API

```bash
curl -X POST "http://<BOARD_IP>:8000/api/models/yolov6n/predict" \
  -F "file=@bus.jpg" -F "conf=0.25" -F "iou=0.65"
```

| 接口 | 方法 | 用途 |
|---|---|---|
| `/` | GET | 网页预览界面 |
| `/api/models/yolov6n/predict` | POST | 检测结果（JSON） |
| `/api/video_feed` | GET | MJPEG 预览流 |
| `/api/config` | GET / POST | 读取或修改运行时阈值 |
| `/api/video/upload` | POST | 上传源视频 |
| `/api/video/analyze` | POST | 启动离线视频分析 |
| `/api/video/status` | GET | 读取分析进度 |
| `/api/video/download/{filename}` | GET | 下载标注后的结果 |

## 实现说明

- 布局 (a)：HEF 已做 NMS（HPP）——应用解析 post-NMS 张量（逐类紧凑缓冲区、
  `Cx5xD` / `CxDx5` 稠密布局、ragged NMS-by-score 列表），`nms_thresh`
  仅保留参数接口。
- 布局 (b)：HEF 输出九个原始分支头（每个 stride 32/16/8 各 4 通道框、1 通道
  objectness、80 通道类别）——应用解码格心到四边的距离（stride 单位），
  用 sigmoid(类别) × sigmoid(objectness) 作为分数，并按 IOU 滑块（Model Zoo
  默认 0.65）做逐类 NMS。
- 预处理把画面 letterbox 到 640x640x3，填充灰色（114），BGR 转 RGB 后送入原始
  uint8；`/255` 归一化已编译进 HEF。
- `cls_id`（0..79）直接索引标准 COCO 类别表；Model Zoo 评测使用
  `labels_offset=1` 对应 COCO 类别 ID。
- 首次推理会打印 HEF 实际返回的布局
  （`[YOLOv6n] layout=..., outputs=[...]`），便于实机核对。

## 硬件验收清单

1. `hailortcli --version` 为 4.23.0 且 `/dev/hailo0` 存在。
2. 启动日志打印 HEF 输入尺寸（`Model input size: ...`），首次推理打印输出布局。
3. 演示视频能显示带类别标签的 COCO 检测框。
4. `POST /api/models/yolov6n/predict` 返回含类别、置信度和框的结果。
5. USB 摄像头模式预览持续刷新，无卡死帧。

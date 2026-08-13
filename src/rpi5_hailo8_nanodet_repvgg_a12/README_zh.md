# NanoDet-RepVGG-a12：Raspberry Pi 5 / CM5 + Hailo-8 目标检测

本模块运行 nanodet_repvgg_a12 目标检测（COCO 80 类，YOLOX 架构），使用
**片上 NMS**（Hailo HPP，`meta_arch=yolox`）。虽然名字带"nanodet"，这个变体
基于 YOLOX 架构（base/yolox.yaml），输入 640×640。HEF 在片上完成 NMS，直接输出
已解码的检测结果，应用只解析后 NMS 张量。FastAPI 服务支持图片、视频文件、USB
摄像头、MJPEG 预览和 REST 推理接口。

## 兼容环境

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-8 PCIe（`/dev/hailo0`） |
| 宿主机/运行时 | HailoRT 4.23.x |
| Python | 3.11，aarch64 |
| 输入 | 640x640x3 **BGR**（input_conversion bgr_to_rgb + no-op normalize_in_net） |
| 输出 | 片上 NMS 张量，后 NMS shape 80x5x100 |
| 类别 | 80（COCO，0-indexed；labels_offset=1 仅用于评估） |
| 参数量 | 5.13M |
| 运算量 | 28.23G |
| HEF | Model Zoo v2.19.0，Hailo-8 |

宿主机驱动、固件、`libhailort.so` 和容器内 Python wheel 必须使用相同的
HailoRT 主版本和次版本。

## 构建

在仓库根目录执行：

```bash
sudo docker build -f docker/hailo8/nanodet_repvgg_a12.dockerfile \
  -t nanodet_repvgg_a12:latest \
  src/rpi5_hailo8_nanodet_repvgg_a12
```

## 运行演示视频

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  nanodet_repvgg_a12:latest \
  python web_detection.py --model_path model/nanodet_repvgg_a12.hef --video_path video/test.mp4
```

浏览器打开 `http://<PI_IP>:8000`。使用 USB 摄像头时，挂载 `/dev/video0`，
并把 `--video_path video/test.mp4` 替换为 `--camera_id 0`。

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/nanodet_repvgg_a12/predict" \
  -F "file=@test.jpg"
```

| 接口 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 健康检查 |
| `/api/models/nanodet_repvgg_a12/predict` | POST | 检测结果（JSON） |
| `/api/video_feed` | GET | MJPEG 预览流 |

## 实现说明

- 与 nanodet_repvgg 同为片上 NMS（HPP）后处理（80 类 HPP 解析），但此变体用
  base/yolox.yaml：`meta_arch=yolox`、640×640 输入、`padding_color=114`
  （灰色，YOLOX 惯例）。
- HEF 在片上做 NMS（`device_pre_post_layers: nms=true, hpp=true`）；应用只解析
  后 NMS 张量（按官方 `tf_postproc_nms`），因此 `nms_thresh` 被忽略（保留是为
  API 兼容）。
- 后 NMS 每行是 `[ymin, xmin, ymax, xmax, score]`，归一化到 640×640 letterbox
  输入的 [0,1]；应用还原成像素并反 letterbox 到原图。
- `normalize_in_net` no-op（mean=0/std=1）+ 片上 `input_conversion(bgr_to_rgb)`：
  应用用灰色 (114) padding 做 letterbox，直接喂**原始 uint8 BGR 像素**——不做
  手动归一化，不做 cvtColor。
- HailoRT 以参差的按类列表（NMS-by-score）返回 NMS vstream；解析器同时兼容
  object/dense 布局。首次推理打印原始 type/shape，便于在硬件上核对（SOP §10）。
- 类别映射：`cls_id`(0..79) → 标准 COCO 80 类列表直接索引（labels_offset=1
  仅用于评估；若硬件上标签偏移，按首次推理日志重新推导）。

## 模型来源

Hailo-8 HEF 来自
[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst)
（模型 `nanodet_repvgg_a12`）。

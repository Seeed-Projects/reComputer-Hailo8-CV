# YOLACT RegNetX-1.6GF：Raspberry Pi 5 / CM5 + Hailo-8 实例分割

本模块运行 YOLACT-RegNetX-1.6GF 实例分割（COCO 80 类，UC Davis），后处理在
**CPU 上完成**（`meta_arch=yolact`）。HEF 输出 16 个原始头——原型 mask、框回归、
mask 系数和置信度——应用负责完整解码链路：anchor 生成、SSD 框解码、Fast NMS、
mask 组装（`proto @ coeffs` + sigmoid + crop），全部 1:1 移植自 Hailo Model Zoo。

FastAPI 服务支持图片、视频文件、USB 摄像头、带 mask 叠加的 MJPEG 预览和 REST 推理接口。

## 兼容环境

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-8 PCIe（`/dev/hailo0`） |
| 宿主机/运行时 | HailoRT 4.23.x |
| Python | 3.11，aarch64 |
| 输入 | 512x512x3 RGB（normalize_in_net ImageNet RGB 均值/方差） |
| 输出 | 16 头：proto（128x128x32）+ 5 个尺度 x {bbox 36, mask 288, conf 729} |
| Priors | 49,104 个 anchor（64/32/16/8/4 特征图，每格 9 个） |
| 类别 | 80（COCO，Model Zoo 类别索引顺序） |
| 参数量 | 30.09M |
| 运算量 | 125.34G |
| HEF | Model Zoo v2.19.0，Hailo-8 |

## 构建

```bash
sudo docker build -f docker/hailo8/yolact_regnetx_1_6gf.dockerfile \
  -t yolact_regnetx_1_6gf:latest \
  src/rpi5_hailo8_yolact_regnetx_1_6gf
```

## 运行演示视频

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  yolact_regnetx_1_6gf:latest \
  python web_detection.py --model_path model/yolact_regnetx_1_6gf.hef --video_path video/test.mp4
```

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/yolact_regnetx_1_6gf/predict" \
  -F "file=@test.jpg"
```

| 端点 | 方法 | 说明 |
|---|---|---|
| `/api/models/yolact_regnetx_1_6gf/predict` | POST | 检测结果 + 每目标 mask 面积（JSON） |
| `/api/video_feed` | GET | MJPEG 预览流（mask + 框） |

## 实现说明

- CPU 后处理（`meta_arch=yolact`）：无片上 NMS。检测流水线（49,104 个 anchor
  → 解码 → 每目标 argmax 类别 → Fast NMS，每类 top-200，最多 100 个目标）是
  Model Zoo `instance_segmentation_postprocessing.py` 的 numpy 移植。
- `normalize_in_net`（ImageNet RGB 均值/方差：[123.68, 116.78, 103.94] /
  [58.4, 57.12, 57.38]）；喂原始 uint8 RGB，无需手动归一化。
- 预处理为直接双线性缩放到 512x512（无 letterbox、无填充），与 Model Zoo 对该
  网络使用的 `mobilenet_ssd` 预处理一致。
- Mask：`sigmoid(proto @ coeffs^T)` 后裁剪到各自框内；按类别上色叠加，
  `mask_thresh=0.5`（对应 YAML `mask_threshold`）。
- 类别映射：类别索引（0..79）→ 直接索引 `CLASS_NAMES_COCO`。
- 首次推理会打印所有输出 vstream shape，便于在硬件上对照网络 YAML 核对 16 头布局。

## 模型来源

[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_instance_segmentation.rst)
（模型 `yolact_regnetx_1_6gf`，源：[dbolya/yolact](https://github.com/dbolya/yolact)）。
# RetinaFace MobileNet-v1：Raspberry Pi 5 / CM5 + Hailo-8 人脸检测

本模块运行 RetinaFace MobileNet-v1 人脸检测（单一人脸类别，WIDER FACE 训练），
后处理在 **CPU 上完成**（`meta_arch=retinaface`）。HEF 输出 9 个原始头——3 个
stride（8/16/32），每个尺度有 bbox、置信度和 5 点关键点——应用负责完整解码
链路：anchor 生成（38,640 个）、SSD 框解码、softmax 置信度、贪心 NMS、关键点
解码，全部移植自 Hailo Model Zoo。

FastAPI 服务支持图片、视频文件、USB 摄像头、带框 + 关键点叠加的 MJPEG 预览
和 REST 推理接口。

## 兼容环境

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-8 PCIe（`/dev/hailo0`） |
| 宿主机/运行时 | HailoRT 4.23.x |
| Python | 3.11，aarch64 |
| 输入 | 736x1280x3 BGR（normalize_in_net 均值 [123,117,104]，std 1） |
| 输出 | 9 头：3 个尺度 x {bbox 8ch, conf 4ch, landmark 20ch} |
| Priors | 38,640 个 anchor（每格 2 个，特征图 92x160/46x80/23x40） |
| 类别 | 1（人脸）+ 每个目标 5 个关键点 |
| 参数量 | 3.49M |
| 运算量 | 25.14G |
| HEF | Model Zoo v2.19.0，Hailo-8 |

## 构建

```bash
sudo docker build -f docker/hailo8/retinaface_mobilenet_v1.dockerfile \
  -t retinaface_mobilenet_v1:latest \
  src/rpi5_hailo8_retinaface_mobilenet_v1
```

## 运行演示视频

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  retinaface_mobilenet_v1:latest \
  python web_detection.py --model_path model/retinaface_mobilenet_v1.hef --video_path video/test.mp4
```

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/retinaface_mobilenet_v1/predict" \
  -F "file=@test.jpg"
```

| 端点 | 方法 | 说明 |
|---|---|---|
| `/api/models/retinaface_mobilenet_v1/predict` | POST | 人脸 + 置信度 + 5 关键点（JSON） |
| `/api/video_feed` | GET | MJPEG 预览流（框 + 关键点） |

## 实现说明

- CPU 后处理（`meta_arch=retinaface`）：无片上 NMS。anchor 生成、SSD 解码
  （variances 10/5）、softmax 置信度、贪心 NMS 和关键点解码是 Model Zoo
  `face_detection_postprocessing.py` 的 numpy 移植。
- 头布局按官方 `collect_box_class_predictions`：8 通道头是 bbox（2 anchor x
  4 坐标），4 通道头是置信度（2 anchor x {背景, 人脸}），20 通道头是关键点
  （2 anchor x 10 坐标）。
- `normalize_in_net` BGR 均值 [123, 117, 104]（std 1）；alls 脚本在 HEF 内
  应用了 `input_conversion(bgr_to_rgb)`，所以应用直接喂原始 uint8 **BGR** 帧。
- 预处理：等比缩放 + 右/下填充（颜色 0）到 736x1280，与 Model Zoo 的
  `_ar_preserving_resize_and_crop` 一致。
- 配置默认值：`score_threshold=0.02` / `nms_iou_thresh=0.4`（官方评测值）；
  实时预览默认置信度 0.5，画面更干净。
- 首次推理会打印所有输出 vstream shape，便于在硬件上核对 9 头布局。

## 模型来源

[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_face_detection.rst)
（模型 `retinaface_mobilenet_v1`，源：[biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)）。
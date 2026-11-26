# Face Landmarks Lite - Raspberry Pi 5 / CM5 + Hailo-8

MediaPipe Face Landmarks Lite（0.6M 参数）在 Hailo-8 上运行，使用 YuNet 在
CPU 上检测人脸并裁出 ROI。Hailo HEF 输出的是 **每张脸 468 个关键点**，
不是 98 个点。

> Hailo Model Zoo 只发布关键点 HEF，没有为 Face Landmarks Lite 提供配套的
> 人脸检测 HEF；官方 `face_detection_landmarks` 多网络使用的是
> `lightface_slim + tddfa_mobilenet_v1`。本模块因此用 YuNet 在 CPU 上做人脸
> 检测，Hailo-8 只负责 468 点关键点推理。

## 模型信息

| 项目 | 值 |
|---|---|
| Hailo HEF | v2.20.0 `hailo8/face_landmarks_lite.hef` |
| HEF 大小 | 3,135,094 bytes |
| 输入 | 192x192x3 RGB 人脸裁剪图 |
| 输出 | 1404 个关键点值 + 1 个人脸置信度值 |
| 关键点 | 468 x (x, y, z) |
| 参数量 | 0.6M |
| 运算量 | 0.07G |
| 官方 FPS | 601（batch 1）/ 2317（batch 8） |
| 全精度 NME | 4.28 |
| 人脸检测 | OpenCV YuNet 2023mar，CPU，MIT |
| License | Apache-2.0（MediaPipe 模型） |

## 运行环境

- Python 3.11 aarch64
- HailoRT 4.23.0
- 宿主机驱动、固件、`libhailort.so` 和 Python wheel 的 major.minor 必须一致
- `opencv-python` 需要支持 `FaceDetectorYN`

## 构建与运行

```bash
cd src/rpi5_hailo8_face_landmarks_lite
sudo docker build -f ../../docker/hailo8/face_landmarks_lite.dockerfile \
  -t rpi5-hailo8-face-landmarks:latest .

sudo docker run --rm --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  rpi5-hailo8-face-landmarks:latest
```

浏览器打开 `http://<开发板IP>:8000` 查看 MJPEG 预览。

USB 摄像头模式：

```bash
sudo docker run --rm --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  --device /dev/video0:/dev/video0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  rpi5-hailo8-face-landmarks:latest \
  python web_detection.py --model_path model/face_landmarks_lite.hef --camera_id 0
```

## REST API

```bash
curl -X POST "http://<开发板IP>:8000/api/models/face_landmarks_lite/predict" \
  -F "file=@face.jpg"
```

返回每张脸一个条目，包含人脸框、置信度和 468 个 `{x, y, z}` 关键点。
`conf` 和 `iou` 是可选表单参数。

## 实现说明

- YuNet 在长边不超过 960 的缩放图上检测人脸，再映射回原图。
- 人脸框向外扩 20%，补成正方形后裁剪、缩放到 192x192，再 BGR→RGB。
- HEF 输出 1404 个值；解码器同时兼容归一化 `[0, 1]` 坐标和 192 像素坐标，
  按首帧 x/y 范围自动判断。
- 关键点从 192x192 裁剪图映射回原图；z 保留模型输出的相对深度。
- HEF 第二个输出是人脸置信度；超出 `[0, 1]` 时做 sigmoid，作为
  `landmark_score` 返回。

## 接口

| 接口 | 方法 | 说明 |
|---|---|---|
| `/` | GET | MJPEG 预览 |
| `/api/video_feed` | GET | MJPEG 流 |
| `/api/config` | GET/POST | 人脸置信度、NMS IoU、最大人脸数 |
| `/api/models/face_landmarks_lite/predict` | POST | 每张脸 468 个关键点 |
| `/api/video/upload` | POST | 上传离线分析视频 |
| `/api/video/list` | GET | 已上传视频列表 |
| `/api/video/analyze` | POST | 开始离线分析 |
| `/api/video/status` | GET | 分析进度 |
| `/api/video/download/{filename}` | GET | 下载标注视频 |

## 开发信息

- 源码：`src/rpi5_hailo8_face_landmarks_lite/`
- Dockerfile：`docker/hailo8/face_landmarks_lite.dockerfile`
- HEF：Hailo Model Zoo v2.20.0 `hailo8`，URL/SHA256 见 `model/README.md`
- 人脸检测：OpenCV Zoo YuNet 2023mar，MIT，见 `model/YUNET_LICENSE.txt`
- 测试：`python -m unittest discover -s tests -v`
- 实机验证尚未完成，见 `TEST_REPORT.md`。
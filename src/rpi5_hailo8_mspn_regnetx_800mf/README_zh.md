# MSPN RegNetX-800MF 在 Raspberry Pi 5 / CM5 + Hailo-8 上运行

本模块使用 MSPN（Multi-Stage Pose Network）搭配 RegNetX-800MF 骨干网络，
实现单人 2D 姿态估计（17 个 COCO 关键点）。HEF 输出单个热力图
（64×48×17），应用逐通道取 argmax 得到关键点坐标，缩放到输入空间并绘制
COCO 骨架。FastAPI 服务支持图片、视频文件、USB 摄像头、MJPEG 预览和
REST 推理接口。

## 兼容性

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-8 PCIe（`/dev/hailo0`） |
| HailoRT 主机/运行时 | 4.23.x |
| Python | 3.11，aarch64 |
| 输入 | 256x192x3 RGB（normalize_in_net，ImageNet RGB 均值/标准差） |
| 输出 | 热力图 64x48x17（17 个 COCO 关键点） |
| 关键点 | 17 个（鼻、眼、耳、肩、肘、腕、髋、膝、踝） |
| 参数量 | 7.17M |
| 运算量 | 2.94G |
| HEF | Model Zoo v2.19.0，Hailo-8 |

## 构建

```bash
sudo docker build -f docker/hailo8/mspn_regnetx_800mf.dockerfile \
  -t cm5-hailo8-mspn-regnetx-800mf:latest \
  src/rpi5_hailo8_mspn_regnetx_800mf
```

## 运行

```bash
sudo docker run --rm --privileged --net=host \
  --name cm5-hailo8-mspn-regnetx-800mf \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  cm5-hailo8-mspn-regnetx-800mf:latest \
  python web_detection.py --model_path model/mspn_regnetx_800mf.hef --video_path video/test.mp4
```

USB 摄像头模式增加 `--device /dev/video0:/dev/video0`，并以
`--camera_id 0` 运行。Web 预览：`http://<PI_IP>:8000`。

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/mspn_regnetx_800mf/predict" \
  -F "file=@test.jpg"
```

| 接口 | 方法 | 说明 |
|---|---|---|
| `/api/models/mspn_regnetx_800mf/predict` | POST | 17 个关键点（JSON） |
| `/api/video_feed` | GET | 带骨架叠加的 MJPEG 预览 |

## 实现说明

- **单人姿态**：模型假设人物位于裁剪区域中心。多人场景需要先用检测器
  （如 YOLO）裁剪出每个人，再逐人运行 MSPN。
- **后处理**：逐通道 argmax → 64×48 空间坐标 → 缩放到 256×192 输入 →
  逆 letterbox 还原到原图。Model Zoo 参考实现额外做 5×5 高斯模糊和
  四分之一像素偏移；本演示未启用（预览足够，且与本仓库其他姿态模块
  保持一致）。
- **输出布局**：首次推理会打印一次原始 vstream 布局并在实机核实；解析器
  同时支持 (1, 64, 48, 17) 和 (1, 17, 64, 48)。
- `normalize_in_net` 含 ImageNet RGB 均值/标准差；无 input_conversion →
  letterbox 后直接输入原始 uint8 RGB。

## 模型来源

[Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)
（模型 `mspn_regnetx_800mf`，v2.19.0）。
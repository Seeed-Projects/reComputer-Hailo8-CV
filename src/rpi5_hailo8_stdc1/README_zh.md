# STDC1：Raspberry Pi 5 / CM5 + Hailo-8 实时语义分割

本模块使用 STDC1 实现 Cityscapes 19 类实时语义分割。FastAPI 服务支持图片、
视频文件、USB 摄像头、MJPEG 预览和 REST 推理接口。

## 兼容环境

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-8 PCIe（`/dev/hailo0`） |
| 宿主机/运行时 | HailoRT 4.23.x |
| Python | 3.11，aarch64 |
| 输入 | 1024x1920x3 RGB |
| 输出 | Cityscapes 19 类分割掩码 |
| 参数量 | 8.27M |
| HEF | Model Zoo v2.19.0，Hailo-8 |

宿主机驱动、固件、`libhailort.so` 和容器内 Python wheel 必须使用相同的
HailoRT 主版本和次版本。

## 构建

在仓库根目录执行：

```bash
sudo docker build -f docker/hailo8/stdc1.dockerfile \
  -t stdc1:latest \
  src/rpi5_hailo8_stdc1
```

## 运行演示视频

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  stdc1:latest \
  python web_detection.py --model_path model/stdc1.hef --video_path video/test.mp4
```

浏览器打开 `http://<PI_IP>:8000`。使用 USB 摄像头时，挂载 `/dev/video0`，
并把 `--video_path video/test.mp4` 替换为 `--camera_id 0`。

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/stdc1/predict" \
  -F "file=@test.jpg"
```

| 接口 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 健康检查 |
| `/api/models/stdc1/predict` | POST | 分割掩码（JSON） |
| `/api/models/stdc1/visualize` | POST | 叠加结果图（JPEG） |
| `/api/models/stdc1/classes` | GET | Cityscapes 类别列表 |

## 模型来源

Hailo-8 HEF 来自
[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_semantic_segmentation.rst)。

# BiSeNetV2：Raspberry Pi 5 / CM5 + Hailo-8 语义分割

本模块使用 BiSeNetV2 实现 Cityscapes 19 类语义分割。官方 Hailo-8 HEF 接收
1024×2048 RGB 输入，归一化和 argmax 都在网络内部完成，输出单通道类别 ID 掩码。
FastAPI 服务支持图片、视频文件、USB 摄像头、MJPEG 预览和 REST 推理接口。

## 兼容环境

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-8 PCIe（`/dev/hailo0`） |
| 宿主机/运行时 | HailoRT 4.23.x |
| Python | 3.11，aarch64 |
| 输入 | 1024×2048×3 RGB |
| 输出 | 1024×2048×1 Cityscapes 类别 ID 掩码 |
| 参数量 | 3.36M |
| HEF | Model Zoo v2.20.0，Hailo-8 |

宿主机驱动、固件、`libhailort.so` 和容器内 Python wheel 必须使用相同的
HailoRT 主版本和次版本。

## 构建

在仓库根目录执行：

```bash
sudo docker build -f docker/hailo8/bisenetv2.dockerfile \
  -t bisenetv2:latest \
  src/rpi5_hailo8_bisenetv2
```

## 运行演示视频

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  bisenetv2:latest \
  python web_detection.py --model_path model/bisenetv2.hef --video_path video/test.mp4
```

浏览器打开 `http://<PI_IP>:8000`。使用 USB 摄像头时，挂载 `/dev/video0`，
并把 `--video_path video/test.mp4` 替换为 `--camera_id 0`。

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/bisenetv2/predict" \
  -F "file=@test.jpg"
```

| 接口 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 健康检查 |
| `/api/models/bisenetv2/predict` | POST | 分割掩码（JSON） |
| `/api/models/bisenetv2/visualize` | POST | 叠加结果图（JPEG） |
| `/api/models/bisenetv2/classes` | GET | Cityscapes 类别列表 |

## 模型来源

HEF 来自
[Hailo Model Zoo v2.20.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_semantic_segmentation.rst)。
Hailo-8 官方数据为 73.0 float mIoU、72.6 硬件 mIoU、batch 1 下 9.69 FPS；
这些是 Model Zoo 数据，不是本 WebUI 的实测值。

HEF 内置归一化（mean `[83.054, 94.095, 82.187]`，std `[53.856, 54.774, 53.933]`，
RGB 顺序），所以应用直接送原始 RGB uint8 像素。输出已经是 argmax 后的类别 ID
掩码，宿主只负责把 ID 映射到 Cityscapes 调色板。

## 验证状态

静态检查和合成掩码/后处理测试通过，但板端加载、真实 Cityscapes 精度和端到端
FPS 仍待验证。详见 `TEST_REPORT.md`；完成这些检查前，不要把本模块描述为
“硬件验证通过”。
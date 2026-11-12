# ResNet-50 - ImageNet 分类（CM5 + Hailo-8）

ResNet-50 通过 HailoRT 4.23 在 **Hailo-8** 上进行 1000 类 ImageNet 分类。
HEF 接收 224x224 RGB 图像并输出 1000 个分数；主机侧做 softmax，对概率做时间
平滑后在 MJPEG 预览上叠加 Top-K 面板。

## 模型

| 属性 | 值 |
|---|---|
| 架构 | ResNet-50（BRECQ/ImageNet） |
| 输入 | 224x224x3 RGB（直接 resize；ImageNet 归一化已编译进 HEF） |
| 输出 | 1000 类 ImageNet-1k 分数 |
| 参数量 | 25.53M |
| 运算量 | 6.98G |
| Top-1 精度（浮点 / Hailo-8） | 75.2% / 74.7% |
| HEF | Hailo Model Zoo v2.20.0，Hailo-8（17,978,280 字节） |
| 许可证 | MIT（BRECQ） |

## 快速开始

运行基线：Python 3.11 与 HailoRT 4.23。在仓库根目录执行构建命令。

```bash
docker build -t resnet_v1_50 -f docker/hailo8/resnet_v1_50.dockerfile src/rpi5_hailo8_resnet_v1_50

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23:/usr/lib/libhailort.so.4.23:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo8-cv/resnet_v1_50:latest \
  python web_detection.py --model_path model/resnet_v1_50.hef --video_path video/test.mp4
```

打开 `http://<开发板IP>:8000` 查看实时预览。默认演示视频是 Hailo 官方的
`example.mp4`（来自 `hailo-csdata` 资源桶，属于 MIT 许可的 `hailo-apps` 演示素材）。

USB 摄像头需加 `--device /dev/video0:/dev/video0`，并把 `--video_path ...` 改为
`--camera_id 0`。

## API

| 端点 | 方法 | 说明 |
|---|---|---|
| `/` | GET | Web 预览 |
| `/api/video_feed` | GET | MJPEG 流 |
| `/api/models/resnet_v1_50/predict` | POST | Top-K 分类结果（JSON） |

```bash
curl -X POST "http://<开发板IP>:8000/api/models/resnet_v1_50/predict" \
  -F "file=@test.jpg" -F "conf=0.30" -F "iou=5"
```

`iou` 字段沿用为 Top-K 数量。

## 实现说明

- HEF 输出单个 1000 维分数向量；编译时关闭了 `softmax` 和 `argmax`，主机侧做
  softmax 和 Top-K。
- 预处理是直接 resize 到 224x224 并 BGR→RGB；ImageNet mean/std 归一化已编译进
  HEF，不使用 letterbox。
- 实时预览对概率向量做平滑（0.6 上一帧 + 0.4 当前帧），然后绘制 Top-K 面板。
- 启动时从 HEF 读取输入尺寸（224x224）。

## 来源

HEF 来自 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)
（v2.20.0，Hailo-8）：

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/resnet_v1_50.hef
```

演示视频：`https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/video/example.mp4`
（来自 `hailo-ai/hailo-apps`，MIT）。

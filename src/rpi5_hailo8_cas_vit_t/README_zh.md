# CAS-ViT-T - ImageNet 分类（CM5 + Hailo-8）

CAS-ViT-T 通过 HailoRT 4.23 在 **Hailo-8** 上进行 1000 类 ImageNet 分类。HEF
接收 384×384 BGR 图像并输出 1000 个分数；主机侧做 softmax，对概率做时间平滑后
在 MJPEG 预览上叠加 Top-K 面板。

## 模型

| 属性 | 值 |
|---|---|
| 架构 | CAS-ViT-T（Convolutional Additive Self-attention Vision Transformer） |
| 输入 | 384×384×3 BGR（直接 resize；bgr_to_rgb 与 ImageNet 归一化已编译进 HEF） |
| 输出 | 1000 类 ImageNet-1k 分数 |
| 参数量 | 21.76M |
| 运算量 | 20.85G |
| Top-1 精度（浮点 / Hailo-8） | 81.9% / 81.6% |
| 吞吐（Hailo-8，batch 1） | 31.6 FPS |
| HEF | Hailo Model Zoo v2.20.0，Hailo-8（26,429,200 字节） |
| 许可证 | MIT（[Tianfang-Zhang/CAS-ViT](https://github.com/Tianfang-Zhang/CAS-ViT)） |

## 快速开始

运行基线：Python 3.11 与 HailoRT 4.23.0。在仓库根目录执行构建命令。

```bash
docker build -t cas_vit_t -f docker/hailo8/cas_vit_t.dockerfile src/rpi5_hailo8_cas_vit_t

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  cas_vit_t
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
| `/api/models/cas_vit_t/predict` | POST | Top-K 分类结果（JSON） |

```bash
curl -X POST "http://<开发板IP>:8000/api/models/cas_vit_t/predict" \
  -F "file=@test.jpg" -F "conf=0.30" -F "iou=5"
```

响应示例：

```json
{
  "success": true,
  "source": "uploaded image",
  "predictions": [
    {"class": "sports car, sport car", "label": "sports car", "confidence": 0.87}
  ],
  "image": {"width": 1280, "height": 720}
}
```

`iou` 在这里复用为 Top-K 数量，与其他模块的 API 参数保持一致。

## 实现说明

- HEF 只输出一个 1000 维分数向量；ALLS 配方没有 `softmax` 或 `argmax`，因此由
  主机执行 softmax 和 Top-K。若某个 HEF 已经输出归一化概率，概率启发式判断也
  会直接接受。
- 预处理只做 384×384 的直接 resize，并保持 **BGR**；CAS-ViT 的 ALLS 配方会在
  HEF 内执行 `bgr_to_rgb` 和 ImageNet 均值/方差归一化。不要转成 RGB，也不使用
  letterbox。
- 实时预览对概率向量做平滑（0.6 上一帧 + 0.4 当前帧），避免标签闪烁，然后绘制
  Top-K 面板。
- Top-K 数量和置信度阈值可在网页（`/api/config`）中调整。

## 来源

HEF 来自 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) v2.20.0
（Hailo-8）：

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/cas_vit_t.hef
```

演示视频：`https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/video/example.mp4`
（来自 `hailo-ai/hailo-apps`，MIT）。
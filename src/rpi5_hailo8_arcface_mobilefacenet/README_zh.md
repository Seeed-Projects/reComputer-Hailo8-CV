# ArcFace MobileFaceNet on Raspberry Pi 5 + Hailo-8

本模块在 Hailo-8 加速器上运行 ArcFace + MobileFaceNet，输出 512 维人脸特征
向量。应用结构复用 CenterPose 模板（FastAPI 服务、MJPEG 预览、离线视频分析、
USB 摄像头），只替换 ArcFace 的预处理和单输出后处理。

人脸比对由客户端完成：服务只返回上传图片的特征向量，不保存人脸库。

## 兼容性

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-8 PCIe（`/dev/hailo0`） |
| HailoRT 宿主/运行时 | 4.23.x |
| Python | 3.11, aarch64 |
| 输入 | 112×112×3 RGB（letterbox，uint8） |
| 输出 | 512 维人脸特征向量 |
| 参数量 | 2.04M（Hailo Model Zoo） |
| HEF | Model Zoo v2.19.0，Hailo-8 |

宿主机驱动、固件、`libhailort.so` 与容器内 Python wheel 的 HailoRT 主次版本
必须一致。

## 构建

在仓库根目录执行：

```bash
sudo docker build -f docker/hailo8/arcface_mobilefacenet.dockerfile \
  -t arcface_mobilefacenet:latest \
  src/rpi5_hailo8_arcface_mobilefacenet
```

## 运行内置演示视频

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  arcface_mobilefacenet:latest \
  python web_detection.py \
    --model_path model/arcface_mobilefacenet.hef \
    --video_path video/test.mp4
```

浏览器打开 `http://<开发板IP>:8000`。

使用 USB 摄像头时挂载 `/dev/video0`，并把 `--video_path ...` 换成
`--camera_id 0`。

## REST API

```bash
curl -X POST "http://<开发板IP>:8000/api/models/arcface_mobilefacenet/predict" \
  -F "file=@face.jpg"
```

```json
{
  "success": true,
  "source": "uploaded image",
  "embedding": [0.023, -0.156, 0.089, "..."],
  "dimension": 512,
  "image": { "width": 1280, "height": 720 }
}
```

同一个 Web 界面还提供 MJPEG 预览、上传视频离线分析和摄像头模式。

## 实现说明

- `preprocess_frame()` 将画面 letterbox 到 112×112 并 BGR 转 RGB；归一化已经
  编译进 HEF，宿主侧不再做缩放。
- `post_process_hailo()` 优先按名称选取 `fc1` 输出 vstream，返回 512 维 float
  向量。
- 首次推理会打印全部 vstream 名称和 shape，便于在实机上核对输出映射。
- `--class_path` 仅为兼容命令行保留，ArcFace 没有类别表，参数会被忽略。
- 使用余弦相似度做比对时，特征向量由客户端做 L2 归一化。

## 故障排查

| 现象 | 检查项 |
|---|---|
| 提示 `HailoRT is not available` | `hailort-packages/` 中的 wheel 是否与宿主机 HailoRT 版本匹配 |
| 固件版本错误 | 宿主机驱动、`libhailort.so`、容器 wheel 必须都是 4.23.x |
| 特征向量几乎不变 | 确认首次推理打印的 vstream 是 `fc1`，且输入为 RGB |
| 同一张脸相似度偏低 | letterbox 后人脸过小，建议先裁剪人脸再送入 |
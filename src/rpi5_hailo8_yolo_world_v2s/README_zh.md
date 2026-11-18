# 基于 Raspberry Pi 5 + Hailo-8 的 YOLO-World v2s

面向 **Raspberry Pi 5 / CM5 + Hailo-8**（reComputer R 系列）的开放词表目标检测模块。
它运行 Hailo Model Zoo 的 YOLO-World v2s HEF，网络有两个输入：640×640 RGB 图像和
80×512 的 CLIP 文本嵌入矩阵。仓库内置的词表是 80 个 COCO 类别。

> **Hailo-8 支持状态。** Hailo 公开的 `HAILO8_*` 模型表中没有 zero-shot object
> detection，当前 Model Zoo 的网络定义声明
> `supported_hw_arch: [hailo15h, hailo10h]`。本模块使用的 HEF 位于官方
> v2.20.0 `hailo8` 前缀下，是真实的 HEF v2 文件，但尚未在 Hailo-8 实机上运行。
> 在把结果当作“已支持”之前，请先在开发板上验证。Hailo 没有发布 Hailo-8 的
> `yolo_world_v2m.hef`，因此本模块只提供 S 变体。

---

## 1. 功能

- MJPEG 实时预览，绘制检测框、类别和置信度。
- 支持图像、视频抽帧和实时摄像头帧的 REST 预测。
- 离线视频批量分析，使用 libx264 ultrafast 编码。
- `/api/config` 可调整显示用的置信度和 IoU 阈值。
- 从 `model/coco_text_embeddings.npy` 和 `model/coco_classes.txt` 加载 80 槽词表。

HEF 内置 YOLO-World 的文本对比头和 host 侧 NMS 后处理。应用在启动时加载一次
CLIP 文本嵌入，后续每帧复用。

---

## 2. 主机准备（Raspberry Pi 5）

容器访问芯片前，宿主机必须安装 Hailo-8 PCIe 驱动、固件和用户态工具。

```bash
sudo apt update
sudo apt install hailo-all
sudo reboot

hailortcli fw-control identify
ls /dev/hailo0
```

安装 Docker：

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh --mirror Aliyun
sudo systemctl enable docker
sudo systemctl start docker
```

容器内 `hailort` wheel 的主次版本必须与宿主机固件和 `libhailort.so` 的主次版本一致。
仓库内置的 wheel 是 `hailort-4.23.0-cp311-cp311-linux_aarch64.whl`。

---

## 3. 构建与运行

```bash
cd src/rpi5_hailo8_yolo_world_v2s
sudo docker build -f ../../docker/hailo8/yolo_world_v2s.dockerfile \
  -t rpi5-hailo8-yolo-world-v2s:latest .

sudo docker run --rm --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  rpi5-hailo8-yolo-world-v2s:latest
```

默认命令循环播放 `video/test.mp4`，Web UI 地址为 `http://<开发板IP>:8000`。

摄像头模式（覆盖默认命令）：

```bash
sudo docker run --rm --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  --device /dev/video0:/dev/video0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  rpi5-hailo8-yolo-world-v2s:latest \
  python web_detection.py --model_path model/yolo_world_v2s.hef --camera_id 0
```

请把 `4.23.0` 替换为 `hailortcli fw-control identify` 报告的版本。

---

## 4. 模型资产

| 文件 | 大小 | 来源 |
|---|---:|---|
| `model/yolo_world_v2s.hef` | 22,431,677 B | [Model Zoo v2.20.0 hailo8](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/yolo_world_v2s.hef) |
| `model/coco_text_embeddings.npy` | 163,968 B | CLIP ViT-B/32 文本嵌入，形状 `(1, 80, 512)` |
| `model/coco_classes.txt` | 701 B | 80 个 COCO 类别，每行一个 |

HEF 的 SHA256 为
`39b5b17df32c11503aa9a9b5d7d51ad0ee77265b0fb799a73aa01857bb33c6ec`。
完整来源与注意事项见 [model/README.md](model/README.md)。

---

## 5. 词表

YOLO-World HEF 固定接收 80 条文本提示。`coco_text_embeddings.npy` 的第 `i` 行对应
输出类别 `i`，`coco_classes.txt` 的第 `i` 行是该类别在画面上显示的名称。

内置嵌入使用 `openai/clip-vit-base-patch32`（`CLIPTextModelWithProjection`，仅类别名，
L2 归一化）生成，与 Hailo 的 `create_coco_tfrecord_with_text_embeddings.py` 一致。

如需其他词表，请预先计算同样形状的 80 条嵌入，并同时传入两个文件：

```bash
python web_detection.py \
  --model_path model/yolo_world_v2s.hef \
  --embeddings_path /path/to/my_embeddings.npy \
  --class_path /path/to/my_classes.txt
```

未使用的槽位仍需填入有效提示；全零行不是安全的“空类别”，文本对比头可能给出
虚假的中等置信度。请用有意义的提示补满 80 行。

---

## 6. HTTP 接口

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/` | Web UI |
| GET/POST | `/api/config` | 读取/更新显示阈值 |
| POST | `/api/video/upload` | 上传视频 |
| GET | `/api/video/list` | 列出上传与输出 |
| POST | `/api/video/analyze` | 启动离线分析 |
| GET | `/api/video/status` | 进度和错误状态 |
| GET | `/api/video/download/{filename}` | 下载标注 MP4 |
| POST | `/api/models/yolo_world/predict` | 单帧预测 |

示例：

```bash
curl -X POST "http://<开发板IP>:8000/api/models/yolo_world/predict" \
  -F "file=@frame.jpg"
```

返回结果包含每个检测的类别、置信度和像素坐标。

---

## 7. 实现说明

- 输入：`yolo_world_v2s/input_layer1`（640×640×3 RGB，UINT8）和
  `yolo_world_v2s/input_layer2`（80×512 float32 CLIP 嵌入）。输入名称在启动时从
  HEF 读取，不按位置硬编码。
- 图像先 letterbox 到 640×640，再转 RGB；不做手工归一化，HEF 内的 normalization
  层会除以 255。
- 检测框在绘制前会反向映射回原始分辨率。
- HEF 的 NMS 使用 Model Zoo 配置：80 类、IoU 0.7、每类最多 300 个候选、分数阈值
  0.001。界面的置信度滑块只过滤返回结果，不会在 host 侧重新执行 NMS。

---

## 8. 已知限制

- 词表在启动时固定。本模块不在板端运行 CLIP 文本编码器，无法从界面动态修改提示。
- H8 HEF 未列入 Hailo 公开的 H8 模型表；在实机测试通过前，请把本模块视为未验证。
- Hailo 没有发布 H8 的 v2m HEF；M 变体仅在 Hailo-10H 上提供。
- Web 服务面向可信局域网，默认没有身份验证或 TLS。
- 尚未评估逐帧精度；待完成的检查见 `TEST_REPORT.md`。
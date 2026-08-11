# reComputer-Hailo8-CV

[English](./README.md) | [中文]

面向 **Raspberry Pi 5 / CM5 + Hailo-8**（reComputer R 系列）的工业级计算机视觉参考实现。
每个模型是一个独立模块，提供 FastAPI 服务：实时 MJPEG 预览、REST 推理、USB 摄像头、
离线批量视频分析——围绕 PCIe 接口的 Hailo-8 加速器和 HailoRT 4.23.x 构建。

仓库覆盖三类任务——**目标检测**（CenterNet、DAMO-YOLO）、**语义分割**（STDC1）、
**姿态估计**（CenterPose）。所有模块共用同一套骨架（HailoRT 执行器、letterbox 与坐标还原、
帧缓冲、MJPEG 编码），仅预处理、片上后处理映射和解码随 HEF 不同而不同。可把任一模块作为模板
迁移到其他 Hailo Model Zoo 模型，但**后处理必须按真实 HEF 输出重新推导**——不能只换文件名。

---

## 硬件平台

| | |
|---|---|
| 主板 | Raspberry Pi 5 / CM5（reComputer R 系列载板） |
| 加速器 | Hailo-8 M.2（PCIe），设备节点 `/dev/hailo0` |
| 系统 | Raspberry Pi OS Bookworm，内核 6.12+ aarch64 |
| 宿主驱动 | `hailo-all` apt 包（PCIe 驱动、固件、`libhailort.so`） |
| HailoRT | 4.23.x 已验证——宿主驱动 / 固件 / 容器 wheel **主版本.次版本必须一致** |

---

## 已收录模型

| 模型 | 任务 | 参数量 | 模块 | 容器镜像 |
|---|---|---:|---|---|
| CenterPose RegNetX-800MF | 姿态估计（17 个 COCO 关键点） | 12.31M | `src/rpi5_hailo8_centerpose_regnetx_800mf/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/centerpose_regnetx_800mf:latest` |
| STDC1 | 语义分割（Cityscapes 19 类） | 8.27M | `src/rpi5_hailo8_stdc1/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/stdc1:latest` |
| CenterNet (resnet_v1_18) | 目标检测（COCO 80） | 14.22M | `src/rpi5_hailo8_centernet_resnet_v1_18_postprocess/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/centernet_resnet_v1_18_postprocess:latest` |
| CenterNet (resnet_v1_50) | 目标检测（COCO 80） | 30.07M | `src/rpi5_hailo8_centernet_resnet_v1_50_postprocess/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/centernet_resnet_v1_50_postprocess:latest` |
| DAMO-YOLO (tinynasL20_T) | 目标检测（COCO 80） | 11.35M | `src/rpi5_hailo8_damoyolo_tinynas_l20_t/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/damoyolo_tinynas_l20_t:latest` |
| DAMO-YOLO (tinynasL25_S) | 目标检测（COCO 80） | 16.25M | `src/rpi5_hailo8_damoyolo_tinynas_l25_s/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/damoyolo_tinynas_l25_s:latest` |

所有 HEF 来自 [Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo)（Hailo-8 编译目标）。

---

## 快速开始（预构建镜像）

下面以已发布的 DAMO-YOLO (L25_S) 镜像为例，它已内置源码、HailoRT wheel、ffmpeg、`.hef`
和演示视频。宿主只需可用 Hailo 工具链。

### 1. 宿主准备（一次性，在 Pi 上）

#### 安装 Docker

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh --mirror Aliyun
sudo systemctl enable docker
sudo systemctl start docker
```

#### 安装 Hailo 工具链

```bash
sudo apt update
sudo apt install hailo-all
sudo reboot

# 重启后确认芯片，并记录固件版本
hailortcli fw-control identify     # 应输出 4.23.x
ls /dev/hailo0
```

### 2. 运行

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    ghcr.io/seeed-projects/recomputer-hailo8-cv/damoyolo_tinynas_l25_s:latest
```

首次运行会拉取镜像。容器随后循环播放内置 `video/test.mp4`，并在 `8000` 端口提供
Web 界面——浏览器打开 `http://<设备_IP>:8000`。

> **为什么要挂载 `libhailort.so`？** 镜像只含 Python 绑定，原生库必须来自宿主的
> `hailo-all` 包。若固件版本不是 `4.23.0`，把两处 `4.23.0` 替换为
> `hailortcli fw-control identify` 输出的版本（若主.次版本不同，需用匹配的 wheel
> 重新从源码构建镜像）。

### USB 摄像头模式

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    --device /dev/video0:/dev/video0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    ghcr.io/seeed-projects/recomputer-hailo8-cv/damoyolo_tinynas_l25_s:latest \
    python web_detection.py --model_path model/damoyolo_tinynas_l25_s.hef --camera_id 0
```

---

## 仓库结构

```text
reComputer-Hailo8-CV/
├── .github/workflows/build-ghcr-images.yml   # 按模型构建 GHCR（只重建有改动的模型）
├── docker/hailo8/
│   ├── centerpose_regnetx_800mf.dockerfile
│   ├── stdc1.dockerfile
│   ├── centernet_resnet_v1_18_postprocess.dockerfile
│   ├── centernet_resnet_v1_50_postprocess.dockerfile
│   ├── damoyolo_tinynas_l20_t.dockerfile
│   └── damoyolo_tinynas_l25_s.dockerfile
└── src/
    ├── rpi5_hailo8_centerpose_regnetx_800mf/
    ├── rpi5_hailo8_stdc1/
    ├── rpi5_hailo8_centernet_resnet_v1_18_postprocess/
    ├── rpi5_hailo8_centernet_resnet_v1_50_postprocess/
    ├── rpi5_hailo8_damoyolo_tinynas_l20_t/
    └── rpi5_hailo8_damoyolo_tinynas_l25_s/

# 每个模块结构相同（共用骨架）：
src/rpi5_hailo8_<slug>/
    ├── web_detection.py            # FastAPI + 推理/编码线程流水线
    ├── py_utils/
    │   ├── hailo_executor.py        # HailoRT 封装，长生命周期 InferVStreams
    │   └── coco_utils.py           # letterbox + 框/mask 坐标还原
    ├── model/<slug>.hef             # Hailo-8 HEF（内置）
    ├── hailort-packages/            # HailoRT wheel（内置）
    ├── video/test.mp4               # 内置演示视频
    ├── requirements.txt
    ├── README.md / README_zh.md     # 模块详解：构建、命令行、故障排查
    └── TEST_REPORT.md               # 验证记录
```

---

## 从源码构建

用于定制——换 `.hef`、改代码，或针对不同 HailoRT 版本重新构建：

```bash
git clone https://github.com/Seeed-Projects/reComputer-Hailo8-CV.git
cd reComputer-Hailo8-CV/src/rpi5_hailo8_damoyolo_tinynas_l25_s

sudo docker build -f ../../docker/hailo8/damoyolo_tinynas_l25_s.dockerfile \
    -t hailo8-damoyolo-l25s:latest .

# 同样的运行命令，把 ghcr.io 镜像名换成本地标签
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    hailo8-damoyolo-l25s:latest
```

---

## REST API

所有接口监听容器 `8000` 端口；用 `--net=host` 时可通过 `http://<设备_IP>:8000` 访问。
把 `<slug>` 替换为上表中的模型 slug（如 `damoyolo_tinynas_l25_s`）。

| 接口 | 方法 | 用途 |
|---|---|---|
| `/api/models/<slug>/predict` | POST | 对上传图片、指定视频帧或当前摄像头帧做单次推理 |
| `/api/video_feed` | GET | 叠加结果的 MJPEG 实时流（嵌入 `<img>`） |
| `/api/config` | GET / POST | 读取或更新 `obj_thresh` / `nms_thresh` |
| `/api/video/upload` | POST | 上传视频用于批量分析 |
| `/api/video/analyze` | POST | 启动离线分析任务（表单字段 `filename=...`） |
| `/api/video/status` | GET | 查询任务进度 |
| `/api/video/list` | GET | 列出已上传源文件和已完成输出 |
| `/api/video/download/{filename}` | GET | 下载标注后的输出 |

### 推理示例

```bash
# 上传图片
curl -X POST http://<设备_IP>:8000/api/models/damoyolo_tinynas_l25_s/predict -F "file=@test.jpg"

# 上传视频的指定帧（timestamp 单位为秒）
curl -X POST http://<设备_IP>:8000/api/models/damoyolo_tinynas_l25_s/predict \
    -F "video=@test.mp4" -F "timestamp=5.5"

# 当前摄像头帧
curl -X POST http://<设备_IP>:8000/api/models/damoyolo_tinynas_l25_s/predict -F "realtime=true"

# 单次调用覆盖阈值
curl -X POST http://<设备_IP>:8000/api/models/damoyolo_tinynas_l25_s/predict \
    -F "file=@test.jpg" -F "conf=0.5" -F "iou=0.4"
```

检测响应：

```json
{
  "success": true,
  "source": "uploaded image",
  "predictions": [
    {
      "class": "car",
      "confidence": 0.91,
      "box": { "x1": 100, "y1": 120, "x2": 320, "y2": 520 }
    }
  ],
  "image": { "width": 1280, "height": 720 }
}
```

把实时流嵌入任意 HTML 页面：

```html
<img src="http://<设备_IP>:8000/api/video_feed">
```

### 动态阈值更新

```bash
# 读取当前值
curl http://<设备_IP>:8000/api/config
# {"obj_thresh":0.25,"nms_thresh":0.45}

# 更新（任一字段可选）
curl -X POST http://<设备_IP>:8000/api/config \
     -H "Content-Type: application/json" \
     -d '{"obj_thresh":0.4}'
```

> 部分 HEF 在片上做 NMS / 峰值查找（CenterNet `max_finder`），另一些只在片上做
> sigmoid、解码在 CPU（DAMO-YOLO `nanodet_split`）。无论 NMS 在哪里执行，阈值始终
> 对最终输出起过滤作用。

---

## 适配其他模型

把 `src/` 下任一模块作为模板：

1. 复制目录并改名（如 `rpi5_hailo8_<新_slug>/`）。
2. 把新 `.hef` 放入 `model/`，重命名为 `<slug>.hef`（小写，GHCR 镜像名必须小写）。
3. 添加 `docker/hailo8/<slug>.dockerfile`，并在
   `.github/workflows/build-ghcr-images.yml` 加一个 matrix 项。
4. **按真实 HEF 输出重新推导后处理**——首次推理列出 vstream 名称/shape，按名称映射 head
   （当两个输出 shape 相同时不能只凭 shape 猜），并确认 RGB/BGR 与归一化。
   Model Zoo 每个模型的 YAML 文档了输出布局。
5. 更新模块的 `README*.md` 与 `TEST_REPORT.md`。

`docs/CM5_HAILO8_MODEL_DEVELOPMENT_SOP_zh.md` 中的开发 SOP 给出了完整清单
（事前核查、HailoRT 基线、Docker、CI、AI Lab）。

---

## 文档

各模块详解（构建、命令行、故障排查、硬件验证）：

- [CenterPose RegNetX-800MF](src/rpi5_hailo8_centerpose_regnetx_800mf/README_zh.md) — [English](src/rpi5_hailo8_centerpose_regnetx_800mf/README.md)
- [STDC1](src/rpi5_hailo8_stdc1/README_zh.md) — [English](src/rpi5_hailo8_stdc1/README.md)
- [CenterNet (resnet_v1_18)](src/rpi5_hailo8_centernet_resnet_v1_18_postprocess/README_zh.md) — [English](src/rpi5_hailo8_centernet_resnet_v1_18_postprocess/README.md)
- [CenterNet (resnet_v1_50)](src/rpi5_hailo8_centernet_resnet_v1_50_postprocess/README_zh.md) — [English](src/rpi5_hailo8_centernet_resnet_v1_50_postprocess/README.md)
- [DAMO-YOLO (tinynasL20_T)](src/rpi5_hailo8_damoyolo_tinynas_l20_t/README_zh.md) — [English](src/rpi5_hailo8_damoyolo_tinynas_l20_t/README.md)
- [DAMO-YOLO (tinynasL25_S)](src/rpi5_hailo8_damoyolo_tinynas_l25_s/README_zh.md) — [English](src/rpi5_hailo8_damoyolo_tinynas_l25_s/README.md)

验证记录：每个模块带 `TEST_REPORT.md`，硬件项在实机验证后填写。

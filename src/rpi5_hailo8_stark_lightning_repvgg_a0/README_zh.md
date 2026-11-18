# 基于 Raspberry Pi 5 + Hailo-8 的 STARK-Lightning 单目标跟踪

面向 **Raspberry Pi 5 / CM5 + Hailo-8**（reComputer R 系列）的单目标跟踪模块。
它运行 Hailo Model Zoo 官方的 `stark_lightning_repvgg_a0` HEF，网络有两个输入：
128×128 的模板裁剪图和 320×320 的搜索裁剪图。

与检测模块不同，跟踪是有状态的：第一帧用用户框选的初始框生成模板；之后每帧围绕
上一帧的预测裁剪搜索区域，运行 HEF，再把输出的 4 个归一化坐标映射回原图。

---

## 1. 主机准备（Raspberry Pi 5）

```bash
sudo apt update
sudo apt install hailo-all
sudo reboot

hailortcli fw-control identify
ls /dev/hailo0

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh --mirror Aliyun
sudo systemctl enable docker
sudo systemctl start docker
```

仓库内置的 wheel 是 `hailort-4.23.0-cp311-cp311-linux_aarch64.whl`；宿主机固件和
`libhailort.so` 的主次版本必须一致。

---

## 2. 构建与运行

```bash
cd src/rpi5_hailo8_stark_lightning_repvgg_a0
sudo docker build -f ../../docker/hailo8/stark_lightning_repvgg_a0.dockerfile \
  -t rpi5-hailo8-stark:latest .

sudo docker run --rm --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  rpi5-hailo8-stark:latest
```

浏览器打开 `http://<开发板IP>:8000`：

1. 上传视频。
2. 页面显示第一帧，用鼠标拖一个框选中目标。
3. 点击 **Run tracking**，处理完成后播放/下载标注视频。

如果固件不是 `4.23.0`，把两个库路径替换为 `hailortcli fw-control identify` 报告的
版本。

---

## 3. 模型资产

| 文件 | 大小 | 来源 |
|---|---:|---|
| `model/stark_lightning_repvgg_a0.hef` | 4,425,622 B | Model Zoo v2.20.0 hailo8 |
| `hailort-packages/hailort-4.23.0-...whl` | 9,939,376 B | HailoRT 4.23.0 |

HEF SHA256：
`7319ef5e0016e85510a71a86e8231a1fb0558a6475fa041e08919cc960c03299`。

Hailo-8 官方指标：LaSOT Success-plot AUC 56.9（全精度）/ 55.2（硬件），1074 FPS，
4.62M 参数，3.48G 运算。来源与许可见 [model/README.md](model/README.md)。

---

## 4. 跟踪流程

实现与 Model Zoo 的 `tracker_infer.py` 一致：

1. `initialize(frame, box)`：以目标面积的 `2x` 裁剪正方形区域，缩放到 128×128，
   保存为模板。
2. `update(frame)`：以上一帧框中心为中心，按 `5x` 面积裁剪搜索区域，缩放到
   320×320，把模板和搜索图一起送入 HEF。
3. 把输出的 4 个归一化 `xyxy` 乘以搜索裁剪比例、加上裁剪偏移、裁剪到画面范围，
   得到新的 `xywh` 状态。

HEF 内置归一化（mean `[123.675,116.28,103.53]`，std `[58.395,57.12,57.375]`，
RGB 顺序），所以应用直接送原始 RGB 像素的 float32 值即可。

---

## 5. HTTP 接口

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/` | Web UI |
| POST | `/api/video/upload` | 上传视频 |
| GET | `/api/video/list` | 列出上传与输出 |
| GET | `/api/video/first_frame?filename=...` | 返回第一帧 JPEG |
| POST | `/api/video/analyze` | 开始跟踪（`filename`、`x`、`y`、`w`、`h`） |
| POST | `/api/video/analyze/stop` | 请求停止 |
| GET | `/api/video/status` | 进度、输出文件名、错误 |
| GET | `/api/video/download/{filename}` | 下载标注 MP4 |

示例：

```bash
curl -X POST "http://<开发板IP>:8000/api/video/analyze" \
  -F "filename=demo.mp4" -F "x=420" -F "y=180" -F "w=160" -F "h=240"
```

---

## 6. 已知限制

- 必须在第一帧框选目标；不支持中途重新初始化或多目标跟踪。
- 仅处理上传的视频，不提供实时摄像头 MJPEG 预览。
- HEF 是官方 Hailo-8 产物，但本模块尚未在实机运行；硬件验证见
  `TEST_REPORT.md`。
- Web 服务面向可信局域网，默认没有身份验证或 TLS。
- 输出视频不保留音轨；奇数宽高会裁去至多一行或一列像素以适配编码。
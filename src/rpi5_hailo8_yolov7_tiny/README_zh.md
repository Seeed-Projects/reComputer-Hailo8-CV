# 基于 Raspberry Pi 5 + Hailo-8 的 YOLOv7-Tiny

使用 Hailo Model Zoo 官方 Hailo-8 `yolov7_tiny` HEF 进行 COCO 80 类目标检测。NMS 在 HEF
内部执行，应用只解析 NMS 后的结果并绘制检测框。

## 1. 主机准备

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

内置 wheel 为 `hailort-4.23.0-cp311-cp311-linux_aarch64.whl`；宿主机固件和
`libhailort.so` 的主次版本必须一致。

## 2. 构建与运行

```bash
cd src/rpi5_hailo8_yolov7_tiny
sudo docker build -f ../../docker/hailo8/yolov7_tiny.dockerfile -t rpi5-hailo8-yolov7_tiny:latest .

sudo docker run --rm --privileged --net=host   -e PYTHONUNBUFFERED=1   --device /dev/hailo0:/dev/hailo0   -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro   -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro   rpi5-hailo8-yolov7_tiny:latest
```

默认循环播放 `video/test.mp4`，WebUI 地址为 `http://<开发板IP>:8000`。请把
`4.23.0` 替换为 `hailortcli fw-control identify` 报告的版本。

## 3. 模型

| 项目 | 值 |
|---|---|
| HEF | `model/yolov7_tiny.hef`，7,373,091 字节 |
| SHA256 | `a0ae4b66e135d6e4c0b22aea7182c2ebd895cd23e1233e88d18309fa7734a177` |
| 输入 | 640x640x3 |
| 后处理 | 片上 NMS（HPP） |
| COCO mAP（全精度 / 硬件） | 37.1 / 36.3 |
| FPS（batch 1 / batch 8） | 633 / 633 |
| 参数量 / 运算量 | 6.22M / 13.74G |

## 4. HTTP 接口

- `GET /`：WebUI 与 MJPEG 预览。
- `POST /api/models/yolov7/predict`：图像或视频帧预测。
- `POST /api/video/upload`、`POST /api/video/analyze`、`GET /api/video/status`：
  离线视频分析。
- `GET /api/video/download/{filename}`：下载标注 MP4。

## 5. 说明与限制

- HEF 来自官方 Hailo-8 Model Zoo；本开发环境没有 Hailo 硬件，未做实机验证，
  见 `TEST_REPORT.md`。
- NMS 已在 HEF 内完成；界面的 IoU 滑块不参与 NMS。
- Web 服务面向可信局域网，默认没有身份验证或 TLS。
- 输出视频不保留音轨。

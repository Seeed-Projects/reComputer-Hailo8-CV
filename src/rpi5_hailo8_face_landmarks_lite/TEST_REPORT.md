# Hailo-8 Face Landmarks Lite 测试报告

**测试日期**：2026-11-26  
**测试模块**：`src/rpi5_hailo8_face_landmarks_lite/`  
**测试结论**：静态检查与合成后处理校验通过；实机验证待完成

---

## 1. 测试环境

| 项 | 值 |
|---|---|
| 开发主机 | Windows 11，Python 3.12，无 Hailo 设备 |
| 目标硬件 | Raspberry Pi 5 / CM5 + Hailo-8（当前未连接，SSH 超时） |
| HailoRT（目标） | 4.23.0 |
| HEF | `model/face_landmarks_lite.hef`，3,135,094 B |
| HEF SHA256 | `ba11b2fd9eff9a35a2cd088a0b922dc4a1d5bd16e74f8a89188487dd3811169c` |
| 人脸检测 | `model/face_detection_yunet_2023mar.onnx`，232,589 B |
| YuNet SHA256 | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| 测试视频 | `video/test.mp4`，1920x1080，25 FPS |

---

## 2. 已执行的检查

| 检查 | 命令/方法 | 结果 |
|---|---|---|
| Python 语法 | `python -m py_compile web_detection.py py_utils/*.py` | 通过 |
| 468 点解码 | 合成 1404 值输出，检查 shape 和范围 | 通过 |
| 归一化坐标 | 合成 `[0,1]` 坐标，检查缩放到 192 | 通过 |
| 像素坐标 | 合成 0..192 坐标，检查不重复缩放 | 通过 |
| 人脸框扩展 | 边界框扩 20% 并补成正方形、裁剪到画面 | 通过 |
| 坐标映射 | 192x192 关键点映射回原图裁剪框 | 通过 |
| 缺失输出 | 只有 score、没有 1404 值时返回 `None` | 通过 |
| HEF 完整性 | 文件大小与官方 S3 `Content-Length` 一致 | 通过 |
| YuNet 完整性 | 文件大小 232,589 B，SHA256 已记录 | 通过 |

测试命令：

```bash
cd src/rpi5_hailo8_face_landmarks_lite
python -m py_compile web_detection.py py_utils/hailo_executor.py py_utils/landmark_utils.py
python -m unittest discover -s tests -v
```

结果：5 个测试全部通过。

---

## 3. 待完成的实机验证

以下项目必须在连接 Hailo-8 的开发板上执行，当前均为 **Pending**：

- [ ] HEF 能被 HailoRT 4.23.0 加载，打印真实输入/输出 vstream 名称和 shape。
- [ ] 确认第一个 1404 值输出是图像关键点，而不是世界坐标关键点。
- [ ] 确认关键点坐标是 0..192 像素还是 0..1 归一化；首帧日志应打印判断结果。
- [ ] 确认 YuNet CPU 检测在演示视频和 USB 摄像头上的稳定性。
- [ ] 用 `video/test.mp4` 跑通 MJPEG 预览和离线分析。
- [ ] 用 REST `/api/models/face_landmarks_lite/predict` 返回 468 点。
- [ ] 记录端到端 FPS（YuNet + 裁剪 + Hailo HEF）和内存占用。
- [ ] 在真实人脸场景抽查关键点位置、遮挡和侧脸表现。

在这些项目完成前，不得把本模块描述为“实机验证通过”。

---

## 4. 已知风险

1. **HEF 输出顺序**：如果 HEF 同时输出图像关键点 1404 和世界坐标关键点
   1404，当前代码选择第一个最大输出；实机必须用 `hailortcli parse-hef`
   或首帧日志确认。
2. **坐标尺度**：官方 postprocess 对 `face_landmarks_lite` 直接返回原始值，
   没有明确归一化；当前解码器按 x/y 最大值自动判断，实机需核对。
3. **CPU 人脸检测**：YuNet 是 CPU 模型，不是 Hailo HEF；README 和卡片必须
   明确这一点。
4. **H10 现有实现不可复用**：H10 模块按 98 个 2D 点解析，与本模型的
   468 个 3D 点不兼容；不要直接复制。
5. **演示视频人脸较小**：1920x1080 视频中 YuNet 检测到的人脸约十几像素，
   裁剪放大后关键点质量有限；建议用近景摄像头验证。

---

## 5. 复现命令

```bash
cd src/rpi5_hailo8_face_landmarks_lite
sha256sum model/face_landmarks_lite.hef model/face_detection_yunet_2023mar.onnx
python -m unittest discover -s tests -v
```

实机验证：

```bash
sudo docker build -f ../../docker/hailo8/face_landmarks_lite.dockerfile \
  -t rpi5-hailo8-face-landmarks:latest .
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  rpi5-hailo8-face-landmarks:latest
```
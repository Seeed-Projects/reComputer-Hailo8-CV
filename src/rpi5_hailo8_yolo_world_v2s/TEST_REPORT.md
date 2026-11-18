# Hailo-8 YOLO-World v2s 测试报告

**测试日期**：2026-11-18
**测试模块**：[src/rpi5_hailo8_yolo_world_v2s/](./)
**测试结论**：静态检查与合成后处理校验通过；实机验证待完成

---

## 1. 测试环境

| 项 | 值 |
|---|---|
| 开发主机 | Windows 11，Python 3.12，无 Hailo 设备 |
| 目标硬件 | Raspberry Pi 5 / CM5 + Hailo-8（未连接） |
| HailoRT | 宿主机未安装（`hailo_platform` 不可导入） |
| 模型 | `model/yolo_world_v2s.hef`，22,431,677 B |
| HEF SHA256 | `39b5b17df32c11503aa9a9b5d7d51ad0ee77265b0fb799a73aa01857bb33c6ec` |
| 文本嵌入 | `model/coco_text_embeddings.npy`，`(1, 80, 512)` float32 |
| 测试视频 | `video/test.mp4`（随模块复制，未在实机解码） |

---

## 2. 已执行的检查

| 检查 | 命令/方法 | 结果 |
|---|---|---|
| Python 语法 | `python -m py_compile web_detection.py py_utils/hailo_executor.py` | 通过 |
| NMS 对象数组解码 | 合成 80 类 object 数组，类别 0/16/79 各一条 | 通过：按 `ymin,xmin,ymax,xmax` 解码，坐标缩放正确，阈值过滤正确 |
| NMS 稠密数组解码 | 合成 `(1, 80, 100, 5)` float32 | 通过：低分框被过滤，类别正确 |
| 空输出 | 80 类全空 | 通过：返回 `None`，不产生虚假框 |
| letterbox 反映射 | 3840×2160 → 640×640，框 `[100,200,300,500]` | 通过：映射回 `[600,360,1800,2160]` |
| 嵌入形状 | `np.load` | 通过：`(1, 80, 512)`，float32 |
| 嵌入归一化 | 逐行 L2 范数 | 通过：≈1.0 |
| 类别表 | 读取 `coco_classes.txt` | 通过：80 行，第 0/16/79 行为 person/dog/toothbrush |
| Dockerfile CMD | 检查引用的 HEF 路径 | 通过：`model/yolo_world_v2s.hef` 存在 |
| CI 矩阵 | 检查 workflow option/include | 通过：`yolo_world_v2s` 已加入 |

合成校验使用与 `web_detection.py` 相同的 `post_process_hailo()` 和
`preprocess_frame()`，不是对真实 HEF 输出的验证。

---

## 3. 待完成的实机验证

以下项目必须在连接 Hailo-8 的开发板上执行，当前状态均为 **Pending**：

- [ ] HEF 能被 HailoRT 4.23 加载，打印真实的输入/输出 vstream 名称与形状。
- [ ] `input_layer1` 接受 UINT8 640×640×3 RGB，`input_layer2` 接受 float32 80×512。
- [ ] 首帧推理成功，NMS 输出布局与 `post_process_hailo()` 匹配。
- [ ] 80 个 COCO 标签与嵌入行一一对应，无错位。
- [ ] 用 `video/test.mp4` 跑通离线分析和 MJPEG 预览。
- [ ] 记录推理 FPS、端到端 FPS 和内存占用。
- [ ] 在真实场景抽查误检/漏检，记录置信度分布。

在完成上述项目之前，不得把本模块描述为“硬件验证通过”。

---

## 4. Hailo-8 支持风险

Hailo 公开的 Hailo-8 模型文档没有 zero-shot object detection 条目，Model Zoo 的
`yolo_world_v2s.yaml` 声明 `supported_hw_arch: [hailo15h, hailo10h]`。本模块使用的
HEF 位于官方 v2.20.0 `hailo8` 前缀下，文件结构为 HEF v2，但没有官方 H8 文档或
基准数据背书。若实机加载失败，应停止发布该镜像，并等待 Hailo 提供正式的 H8 构建。

Hailo 未发布 Hailo-8 的 `yolo_world_v2m.hef`（已检查 v2.15.0–v2.20.0、v5.4.0、
v5.5.0），因此本仓库不提供 H8 M 变体。

---

## 5. 复现命令

```bash
python -m py_compile web_detection.py py_utils/hailo_executor.py
python - <<'PY'
import numpy as np
emb = np.load("model/coco_text_embeddings.npy")
print(emb.shape, emb.dtype)
print(np.linalg.norm(emb[0], axis=-1).min(), np.linalg.norm(emb[0], axis=-1).max())
PY
```

实机验证需要在 Hailo-8 开发板上运行容器，见 `README.md` 的 Docker 命令。
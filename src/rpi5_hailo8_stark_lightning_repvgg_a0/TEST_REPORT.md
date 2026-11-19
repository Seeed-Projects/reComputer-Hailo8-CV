# Hailo-8 STARK-Lightning 测试报告

**测试日期**：2026-11-18
**测试模块**：[src/rpi5_hailo8_stark_lightning_repvgg_a0/](./)
**测试结论**：静态检查与跟踪数学校验通过；实机验证待完成

---

## 1. 测试环境

| 项 | 值 |
|---|---|
| 开发主机 | Windows 11，Python 3.12，无 Hailo 设备 |
| 目标硬件 | Raspberry Pi 5 / CM5 + Hailo-8（未连接） |
| HailoRT | 宿主机未安装（`hailo_platform` 不可导入） |
| 模型 | `model/stark_lightning_repvgg_a0.hef`，4,425,622 B |
| HEF SHA256 | `7319ef5e0016e85510a71a86e8231a1fb0558a6475fa041e08919cc960c03299` |
| 测试视频 | `video/test.mp4`（随模块复制，未在实机解码） |

---

## 2. 已执行的检查

| 检查 | 命令/方法 | 结果 |
|---|---|---|
| Python 语法 | `python -m py_compile web_detection.py py_utils/hailo_executor.py py_utils/stark_tracker.py` | 通过 |
| 模板裁剪 | 在 1920×1080 合成帧上对已知框调用 `sample_target(..., 2.0, 128)` | 通过：输出 128×128 |
| 搜索裁剪 | 同一帧调用 `sample_target(..., 5.0, 320)` | 通过：输出 320×320 |
| 边界裁剪 | `clip_box` 对越界框 | 通过：结果留在画面内 |
| 坐标映射 | `map_box_back` 把归一化框映射回图像坐标 | 通过：结果与已知状态一致、宽高为正 |
| 盒头解码 | ONNX ReferenceEvaluator：`conv_tower.2` → softmax/soft-argmax 对比 `pred_boxes` | 通过：最大绝对误差 < 1e-7；one-hot 热图映射也通过 |
| HEF 输入/输出 | 实机日志（2026-11-19） | 输入 `input_layer2 (320,320,3)`、`input_layer1 (128,128,3)`；输出 `conv33 (1,20,20,2)` |
| 输入 shape | 桩模块加载执行器，调用 `_as_single_frame` | 通过：模板 `(1,128,128,3)`、搜索 `(1,320,320,3)` |
| HEF 大小/SHA | `sha256sum` | 通过：与 model/README 一致 |
| Dockerfile CMD | 检查引用的 HEF 路径 | 通过：`model/stark_lightning_repvgg_a0.hef` 存在 |
| CI 矩阵 | 检查 workflow option/include | 通过：`stark_lightning_repvgg_a0` 已加入 |

这些检查不覆盖真实 HailoRT 的完整跟踪效果；HEF 的输入/输出形状已由实机日志确认，
盒头解码已用 ONNX 参考输出验证。

---

## 3. 待完成的实机验证

以下项目必须在连接 Hailo-8 的开发板上执行，当前状态均为 **Pending**：

- [x] HEF 能被 HailoRT 4.23 加载，实际输出为 `conv33 (1,20,20,2)`，已由 host 侧 soft-argmax 解码。
- [x] `input_layer1` 接受 `128×128×3` 模板，`input_layer2` 接受 `320×320×3` 搜索图。
- [ ] 首帧初始化后，模板和搜索裁剪的 RGB/归一化与官方预处理一致。
- [ ] 用一个真实视频完成完整跟踪，检查框不会明显漂移或丢失。
- [ ] 记录推理 FPS、端到端 FPS 和内存占用。
- [ ] 抽查遮挡、快速运动和出画场景，记录失败模式。

在完成上述项目之前，不得把本模块描述为“硬件验证通过”。

---

## 4. 复现命令

```bash
python -m py_compile web_detection.py py_utils/hailo_executor.py py_utils/stark_tracker.py
python - <<'PY'
import numpy as np
from py_utils.stark_tracker import sample_target, clip_box, map_box_back
frame = np.zeros((1080, 1920, 3), np.uint8)
for factor, size in ((2.0, 128), (5.0, 320)):
    crop, scale = sample_target(frame, (900, 500, 120, 200), factor, size)
    print(factor, size, crop.shape, round(scale, 4))
print(clip_box((-50, -50, 200, 200), 1080, 1920, margin=10))
PY
```

实机验证需要在 Hailo-8 开发板上运行容器，见 `README.md` 的 Docker 命令。
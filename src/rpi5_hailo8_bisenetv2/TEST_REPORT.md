# Hailo-8 BiSeNetV2 测试报告

**测试日期**：2026-11-24
**测试模块**：[src/rpi5_hailo8_bisenetv2/](./)
**测试结论**：静态检查、合成掩码后处理和预处理校验通过；实机验证待完成

---

## 1. 测试环境

| 项 | 值 |
|---|---|
| 开发主机 | Windows 11，Python 3.12，无 Hailo 设备 |
| 目标硬件 | Raspberry Pi 5 / CM5 + Hailo-8（未连接） |
| HailoRT | 宿主机未安装（`hailo_platform` 不可导入） |
| 容器 wheel | `hailort-4.23.0-cp311-cp311-linux_aarch64.whl` |
| 模型 | `model/bisenetv2.hef`，13,667,807 B |
| HEF SHA256 | `a047033cb53c43e3904fdb75a2aac6e9feeb5ad44a085422c559e258725054b7` |
| HEF 编译版本 | Hailo Model Zoo v2.20.0（hailo8） |
| 测试视频 | `video/test.mp4`（随模块复制，未在实机解码） |

---

## 2. 已执行的检查

| 检查 | 命令/方法 | 结果 |
|---|---|---|
| Python 语法 | `python -m py_compile web_detection.py py_utils/hailo_executor.py py_utils/coco_utils.py` | 通过 |
| 单通道掩码输出 | `post_process_hailo` 输入 `(1,1024,2048,1)` 合成 argmax 掩码 | 通过：输出 `(1024,2048)` uint8，类别 ID 保持为 0 和 13 |
| 多通道 logits 兼容 | `post_process_hailo` 输入 `(1,19,1024,2048)` 合成 logits | 通过：按通道 argmax，输出 `(1024,2048)` |
| 图像预处理 | `preprocess_frame` 在 1920×1080 合成帧上 letterbox | 通过：输出 `(1024,2048,3)` uint8，RGB 顺序 |
| 掩码叠加 | `draw_segmentation_mask` 用 `(1024,2048)` 掩码和 letterbox 信息还原到 1080p 帧 | 通过：输出帧保持 1920×1080，叠加无形状错误 |
| HEF 大小/SHA | `sha256sum` | 通过：与 model/README 一致 |
| Dockerfile CMD | 检查引用的 HEF 路径 | 通过：`model/bisenetv2.hef` 存在 |
| CI 矩阵 | 检查 workflow option/include | 通过：`bisenetv2` 已加入 |

这些检查不覆盖真实 HailoRT 的输入输出布局和实际分割质量。

---

## 3. 官方模型指标

| 指标 | 值 |
|---|---|
| 输入 | 1024×2048×3 |
| 输出 | 1024×2048×1 argmax 类别 ID 掩码 |
| float mIoU | 73.0 |
| Hardware mIoU | 72.6 |
| FPS（batch 1） | 9.69 |
| 参数量 | 3.36M |
| 运算量 | 196.84G |

以上是 Hailo Model Zoo v2.20.0 的公开数据，不是本容器的实测值。

---

## 4. 待完成的实机验证

以下项目必须在连接 Hailo-8 的开发板上执行，当前状态均为 **Pending**：

- [ ] HEF 能被 HailoRT 4.23 加载，打印真实的输入/输出 vstream。
- [ ] 确认实际输出为 `1024×2048×1` 的 argmax 类别掩码。
- [ ] 用真实 Cityscapes 风格图像检查道路、车辆、行人等类别是否合理。
- [ ] 用一个真实视频完成完整分割，检查掩码与画面是否对齐。
- [ ] 记录推理 FPS、端到端 FPS 和内存占用。
- [ ] 抽查低照度、遮挡和运动模糊场景，记录失败模式。

在完成上述项目之前，不得把本模块描述为“硬件验证通过”。

---

## 5. 复现命令

```bash
cd src/rpi5_hailo8_bisenetv2
python -m py_compile web_detection.py py_utils/hailo_executor.py py_utils/coco_utils.py
python - <<'PY'
import numpy as np
import importlib.util, sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
spec = importlib.util.spec_from_file_location('web_detection', 'web_detection.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mask = np.zeros((1, 1024, 2048, 1), np.float32)
mask[0, 100:200, 300:400, 0] = 13
out = mod.post_process_hailo({'out': mask}, 0, 0, 1024, 2048)
print(out.shape, out.dtype, sorted(np.unique(out).tolist()))
assert out.shape == (1024, 2048) and out.dtype == np.uint8
print('OK')
PY
sha256sum model/bisenetv2.hef
```

实机验证需要在 Hailo-8 开发板上运行容器，见 `README.md` 的 Docker 命令。
# Hailo-8 YOLOv7-Tiny 测试报告

**测试日期**：2026-11-19
**测试模块**：[src/rpi5_hailo8_yolov7_tiny/](./)
**测试结论**：静态检查与合成后处理校验通过；实机验证待完成

## 1. 环境

| 项 | 值 |
|---|---|
| 开发主机 | Windows 11，Python 3.12，无 Hailo 设备 |
| 目标硬件 | Raspberry Pi 5 / CM5 + Hailo-8 |
| HailoRT | 宿主机未安装（`hailo_platform` 不可导入） |
| HEF | `model/yolov7_tiny.hef`，7,373,091 B |
| HEF SHA256 | `a0ae4b66e135d6e4c0b22aea7182c2ebd895cd23e1233e88d18309fa7734a177` |

## 2. 已执行的检查

| 检查 | 方法 | 结果 |
|---|---|---|
| Python 语法 | `py_compile web_detection.py py_utils/hailo_executor.py py_utils/coco_utils.py` | 通过 |
| NMS 后处理 | 合成 object 数组与稠密 `(1,80,100,5)` 输出 | 通过：类别、框、阈值过滤正确 |
| HEF 大小/SHA | `sha256sum` | 通过 |
| Dockerfile CMD | 检查引用的 HEF 路径 | 通过：`model/yolov7_tiny.hef` 存在 |

## 3. 待完成的实机验证

- [ ] HEF 能被 HailoRT 4.23 加载，打印真实的输入/输出 vstream。
- [ ] 单帧推理和视频分析能输出合理的 COCO 检测框。
- [ ] 记录推理 FPS、端到端 FPS 和内存占用。
- [ ] 抽查实际场景的漏检/误检。

在完成上述项目之前，不得把本模块描述为“硬件验证通过”。

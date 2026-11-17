# Model directory

Five Hailo-8 YOLOv8 detector HEFs from Hailo Model Zoo v2.20.0 are bundled here.
All of them include the on-chip NMS post-process layer expected by
`post_process_hailo()`.

| File | Size | Source | SHA256 |
|---|---:|---|---|
| `yolov8n.hef` | 5,155,491 bytes | [Model Zoo v2.20.0](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/yolov8n.hef) | `7103302bf5f2bac163f60b3f9436684e85f762405749020f89249101bd606f49` |
| `yolov8s.hef` | 10,432,255 bytes | [Model Zoo v2.20.0](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/yolov8s.hef) | `d3d2c15d28fc38030cdd00ffef6fad030c8a68cddddf421a4f41194077c313f4` |
| `yolov8m.hef` | 30,516,142 bytes | [Model Zoo v2.20.0](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/yolov8m.hef) | `8cf0a6169b4bcecd345c1361db62bfe00d1eeb9c62d287473129e92db38b3986` |
| `yolov8l.hef` | 44,001,333 bytes | [Model Zoo v2.20.0](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/yolov8l.hef) | `2551c3a420e3f228f83b42bec2e56362a350e83d750978880a99b3944656ffb5` |
| `yolov8x.hef` | 62,712,287 bytes | [Model Zoo v2.20.0](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/yolov8x.hef) | `4732ba4cee18dbacaeb4e3d8e3fae98dbcaacbf0aeb298ccfc13e8c408a039eb` |

Do not replace these with Hailo-10H or Hailo-8L HEFs; HEF files are
hardware-specific. The default container CMD uses `yolov8n.hef`; pass
`--model_path model/yolov8l.hef` or `model/yolov8x.hef` for the larger variants.

# Model artifact

`yolov7e6.hef` is the Hailo-8 build from Hailo Model Zoo v2.20.0:

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/yolov7e6.hef
```

- Size: `80,286,337` bytes
- SHA256: `2438f95b9463561ef2a5dedbdd7d2e2c37fb9afb1cf4d5e00172a1bd3d1fa25c`
- Network: YOLOv7e6, COCO 80-class object detection
- Input: 1280x1280x3
- Post-processing: on-chip NMS (HPP); the application only reads the
  post-NMS output
- Model Zoo metrics: 55.4 / 53.7 mAP (full precision / hardware), 11.3 / 15.9 FPS
  (batch 1 / batch 8), 97.20M parameters, 515.12G operations

Do not replace it with a Hailo-10H, Hailo-15H, or Hailo-8L HEF; HEF files are
hardware-specific.

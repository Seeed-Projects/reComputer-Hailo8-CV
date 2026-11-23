# Model artifact

`yolov9c.hef` is the Hailo-8 build from Hailo Model Zoo v2.20.0:

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/yolov9c.hef
```

- Size: `26,971,779` bytes
- SHA256: `7352cb6d3377af52a39993be4f2096d650eef6bdf2e5481a2a692ca2502328a6`
- Network: YOLOv9c, COCO 80-class object detection
- Input: 640x640x3
- Post-processing: on-chip NMS (HPP); the application only reads the
  post-NMS output
- Model Zoo metrics: 52.6 / 51.0 mAP (full precision / hardware),
  44.8 / 88.7 FPS (batch 1 / batch 8), 25.3M parameters, 102.1G operations
- Upstream: <https://github.com/WongKinYiu/yolov9>, GPL-3.0

Do not replace it with a Hailo-10H, Hailo-15H, or Hailo-8L HEF; HEF files are
hardware-specific.

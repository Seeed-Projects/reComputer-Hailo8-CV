# Model artifact

`yolov7_tiny.hef` is the Hailo-8 build from Hailo Model Zoo v2.20.0:

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/yolov7_tiny.hef
```

- Size: `7,373,091` bytes
- SHA256: `a0ae4b66e135d6e4c0b22aea7182c2ebd895cd23e1233e88d18309fa7734a177`
- Network: YOLOv7-Tiny, COCO 80-class object detection
- Input: 640x640x3
- Post-processing: on-chip NMS (HPP); the application only reads the
  post-NMS output
- Model Zoo metrics: 37.1 / 36.3 mAP (full precision / hardware), 633 / 633 FPS
  (batch 1 / batch 8), 6.22M parameters, 13.74G operations

Do not replace it with a Hailo-10H, Hailo-15H, or Hailo-8L HEF; HEF files are
hardware-specific.

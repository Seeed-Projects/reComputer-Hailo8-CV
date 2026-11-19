# Model artifact

`yolov7x.hef` is the Hailo-8 build from Hailo Model Zoo v2.20.0:

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/yolov7x.hef
```

- Size: `63,971,967` bytes
- SHA256: `ab74cf3fec338aaad54d2405caf9e5037a6ce085d8fd179a1b16883fb5d28854`
- Network: YOLOv7x, COCO 80-class object detection
- Input: 640x640x3
- Post-processing: on-chip NMS (HPP); the application only reads the
  post-NMS output
- Model Zoo metrics: 52.4 / 51.0 mAP (full precision / hardware), 31.0 / 58.7 FPS
  (batch 1 / batch 8), 71.46M parameters, 189.68G operations

Do not replace it with a Hailo-10H, Hailo-15H, or Hailo-8L HEF; HEF files are
hardware-specific.

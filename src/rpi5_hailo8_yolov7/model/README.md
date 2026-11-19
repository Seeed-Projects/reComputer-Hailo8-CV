# Model artifact

`yolov7.hef` is the Hailo-8 build from Hailo Model Zoo v2.20.0:

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/yolov7.hef
```

- Size: `38,299,965` bytes
- SHA256: `5991261b5cd6c093b58b1712b3ae4bba9c5c46026783ff0538e2c1d50c76ef94`
- Network: YOLOv7, COCO 80-class object detection
- Input: 640x640x3
- Post-processing: on-chip NMS (HPP); the application only reads the
  post-NMS output
- Model Zoo metrics: 50.6 / 48.8 mAP (full precision / hardware), 54.5 / 99.6 FPS
  (batch 1 / batch 8), 36.91M parameters, 104.51G operations

Do not replace it with a Hailo-10H, Hailo-15H, or Hailo-8L HEF; HEF files are
hardware-specific.

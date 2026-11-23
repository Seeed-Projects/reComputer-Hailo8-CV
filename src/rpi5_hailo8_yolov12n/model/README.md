# Model artifact

`yolov12n.hef` is the Hailo-8 build from Hailo Model Zoo v2.20.0:

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/yolov12n.hef
```

- Size: `8,940,675` bytes
- SHA256: `a1837b80fab4f102beb78c911cbfce2dcf93593d7015abecf2c1fadb2287d251`
- Network: YOLOv12n, COCO 80-class object detection
- Input: 640x640x3
- Post-processing: on-chip NMS (HPP); the application only reads the
  post-NMS output
- Model Zoo metrics: 40.5 / 39.3 mAP (full precision / hardware),
  46.2 / 150 FPS (batch 1 / batch 8), 2.6M parameters, 7.5G operations
- Upstream: <https://github.com/sunsmarterjie/yolov12>, AGPL-3.0

Do not replace it with a Hailo-10H, Hailo-15H, or Hailo-8L HEF; HEF files are
hardware-specific.

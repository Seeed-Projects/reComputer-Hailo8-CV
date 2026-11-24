# Model artifact

`ddrnet_23_slim.hef` is the Hailo-8 build from Hailo Model Zoo v2.20.0:

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/ddrnet_23_slim.hef
```

- Size: `9,724,219` bytes
- SHA256: `a35943e90ae242199d951df3e848f7e774ebc6df86da50a08d708cbf0f7ec269`
- Network: DDRNet-23-Slim, Cityscapes 19-class semantic segmentation
- Input: `1024x2048x3` RGB
- Output: `1024x2048x1` argmax class-ID mask
- Model Zoo metrics: 76.7 float mIoU / 76.3 hardware mIoU, 38.1 FPS at batch 1,
  5.70M parameters, 72.60G operations
- Upstream: <https://github.com/ydhongHIT/DDRNet>, MIT

The HEF contains the in-network normalization used by the Model Zoo YAML:
mean `[123.675, 116.28, 103.53]`, std `[58.395, 57.12, 57.375]`, RGB order.
Feed raw RGB uint8 pixels; the HEF also performs the final argmax.

Do not replace it with a Hailo-10H, Hailo-15H, or Hailo-8L HEF; HEF files are
hardware-specific.
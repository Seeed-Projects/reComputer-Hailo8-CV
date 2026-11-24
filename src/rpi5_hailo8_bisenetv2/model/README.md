# Model artifact

`bisenetv2.hef` is the Hailo-8 build from Hailo Model Zoo v2.20.0:

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/bisenetv2.hef
```

- Size: `13,667,807` bytes
- SHA256: `a047033cb53c43e3904fdb75a2aac6e9feeb5ad44a085422c559e258725054b7`
- Network: BiSeNetV2, Cityscapes 19-class semantic segmentation
- Input: `1024x2048x3` RGB
- Output: `1024x2048x1` argmax class-ID mask
- Model Zoo metrics: 73.0 float mIoU / 72.6 hardware mIoU, 9.69 FPS at batch 1,
  3.36M parameters, 196.84G operations
- Upstream: <https://github.com/CoinCheung/BiSeNet>, MIT

The HEF contains the in-network normalization used by the Model Zoo YAML:
mean `[83.054, 94.095, 82.187]`, std `[53.856, 54.774, 53.933]`, RGB order.
Feed raw RGB uint8 pixels; the HEF also performs the final argmax.

Do not replace it with a Hailo-10H, Hailo-15H, or Hailo-8L HEF; HEF files are
hardware-specific.
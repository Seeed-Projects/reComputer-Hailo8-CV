# Model artifact

## `stark_lightning_repvgg_a0.hef` (Hailo-8)

Published in the official Hailo Model Zoo v2.20.0 `hailo8` prefix:

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/stark_lightning_repvgg_a0.hef
```

- Size: `4,425,622` bytes
- SHA256: `7319ef5e0016e85510a71a86e8231a1fb0558a6475fa041e08919cc960c03299`
- Network: STARK-Lightning, RepVGG-A0 backbone, single-object tracking
- Inputs: `input_layer1` template `128x128x3`, `input_layer2` search `320x320x3`
- HEF output: `conv33`, two 20x20 corner heatmaps `(1, 20, 20, 2)`
- Host decode: per-channel softmax over the 400 cells, soft-argmax on the
  320x320 pixel-center grid (8..312, step 16), concat `[x1,y1,x2,y2]`, divide by
  320; this matches the ONNX `pred_boxes [1, 4]` output

The HEF contains the in-network normalization used by the Model Zoo YAML:
mean `[123.675, 116.28, 103.53]`, std `[58.395, 57.12, 57.375]`, RGB order.
Feed raw RGB pixels (0-255) as float32 host buffers.

## Official metrics

Hailo-8 public single-object-tracking table (LaSOT Success-plot AUC):

| Metric | Value |
|---|---:|
| Float AUC | 56.9 |
| Hardware AUC | 55.2 |
| FPS (batch 1 / batch 8) | 1074 / 1074 |
| Parameters | 4.62M |
| Operations | 3.48G |

Source: Hailo Model Zoo `HAILO8_single_object_tracking.rst` (DFC v2.20.0).
The model is trained on LaSOT + GOT10K + COCO17 + TrackingNet and evaluated on
a partial LaSOT-extension set.

## Provenance and licensing

- Upstream source: <https://github.com/researchmm/Stark>
- License: MIT
- Pretrained archive:
  <https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/SingleObjectTracking/stark_lightning_repvgg_a0/pretrained/2026-08-03/stark_lightning_repvgg_a0.zip>

## Hailo-10H note

Hailo also publishes a Hailo-10H build in Model Zoo v5.5.0
(`stark_lightning_repvgg_a0.hef`, 6,029,312 bytes); the v5.4.0 directory does
not contain it. This module ships the Hailo-8 artifact only.

The current master network YAML lists `supported_hw_arch: [hailo15h,
hailo10h]`, but the v2.20.0 Hailo-8 documentation and the HEF above are the
official Hailo-8 release. Do not mix HEFs between accelerators.
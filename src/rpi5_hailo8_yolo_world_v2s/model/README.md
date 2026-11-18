# Model artifacts

## `yolo_world_v2s.hef` (Hailo-8)

Published in the official Hailo Model Zoo v2.20.0 `hailo8` prefix:

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/yolo_world_v2s.hef
```

- Size: `22,431,677` bytes
- SHA256: `39b5b17df32c11503aa9a9b5d7d51ad0ee77265b0fb799a73aa01857bb33c6ec`
- Network: YOLO-World v2s, zero-shot object detection, 640x640 image input,
  80 x 512 CLIP text-embedding input, host-side NMS post-process
- Model Zoo metadata: 9.3M parameters, 32.7G operations, 36.3 COCO mAP
  (full precision)

**Hailo-8 support caveat.** Hailo's public `HAILO8_*` model tables do not list
zero-shot object detection, and the current Model Zoo network definition
(`hailo_model_zoo/cfg/networks/yolo_world_v2s.yaml`) declares
`supported_hw_arch: [hailo15h, hailo10h]`. The file above is a real HEF v2
object in the official `hailo8` prefix (last modified 2026-10-14), but H8
support is not documented. Treat this module as unverified until it has been
run on a physical Hailo-8.

No official Hailo-8 `yolo_world_v2m.hef` exists in the Model Zoo versions
checked (`v2.15.0` through `v2.20.0`, `v5.4.0`, `v5.5.0`). The H8 module
therefore ships the S variant only; compiling M for Hailo-8 is outside the
official-artifact workflow.

## `coco_text_embeddings.npy`

- Size: `163,968` bytes
- SHA256: `ffeb1fe385ac7cc0beefbdc8dede0eec4384f9212e1a193ed4fcdced3a3281be`
- Shape: `(1, 80, 512)`, float32, L2-normalized per row

The 80 rows are CLIP ViT-B/32 text embeddings for the COCO class names in
`coco_classes.txt`, in the same order. They were generated with
`openai/clip-vit-base-patch32` (`CLIPTextModelWithProjection`, class name only,
no prompt template), matching
`hailo_model_zoo/datasets/create_coco_tfrecord_with_text_embeddings.py`.

## `coco_classes.txt`

- Size: `701` bytes
- SHA256: `4d4aaea7bee6be2f675d9b53a9195ca36dfe6429f7479f29155da522a6c85930`

One COCO class name per line, 80 rows. Row `i` labels output class `i` and must
remain aligned with row `i` of the embedding file.

Do not replace the HEF with a Hailo-10H, Hailo-15H, or Hailo-8L build; HEF
files are hardware-specific. If you export a custom 80-prompt vocabulary,
replace both the embeddings and the class file together.
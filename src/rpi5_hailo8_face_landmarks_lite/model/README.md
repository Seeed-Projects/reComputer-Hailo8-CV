# Model artifacts

## `face_landmarks_lite.hef` (Hailo-8)

Published in the official Hailo Model Zoo v2.20.0 `hailo8` prefix:

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/face_landmarks_lite.hef
```

- Size: `3,135,094` bytes
- SHA256: `ba11b2fd9eff9a35a2cd088a0b922dc4a1d5bd16e74f8a89188487dd3811169c`
- Official table: `HAILO8_facial_landmark_detection.rst` (v2.20.0)
- `supported_hw_arch`: `hailo8`, `hailo8l`
- Input: `192x192x3` RGB
- Output: `1x1x1404` landmarks + `1x1x1` face-presence value
- 468 landmarks x 3 coordinates; 0.6M parameters, 0.07G operations
- Full-precision NME: 4.28

The HEF normalizes the input with mean 0 and std 255 in the network. Feed a
cropped, RGB face image; do not letterbox a full frame and expect usable
landmarks.

Do not replace this with the Hailo-10H HEF. The H10 build is a different
artifact.

## `face_detection_yunet_2023mar.onnx` (CPU face ROI detector)

The Model Zoo table only publishes the landmark model; it does not pair it
with a face detector. This module uses YuNet on the CPU to locate faces and
crops a square ROI before calling the Hailo landmark HEF.

```text
https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

- Size: `232,589` bytes
- SHA256: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`
- License: MIT (see `YUNET_LICENSE.txt`)
- WIDER Face AP: 0.834 easy / 0.824 medium / 0.708 hard

YuNet is a CPU detector, not a Hailo HEF. The Hailo-8 device is used only for
the 468-point landmark inference.
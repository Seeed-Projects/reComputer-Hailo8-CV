# ResNet-50 - ImageNet Classification on CM5 + Hailo-8

ResNet-50 performs 1000-class ImageNet classification on **Hailo-8** through
HailoRT 4.23. The HEF takes a 224x224 RGB frame and returns 1000 scores; the
host applies softmax, smooths the probabilities over time and overlays a Top-K
panel on the MJPEG preview.

## Model

| Property | Value |
|---|---|
| Architecture | ResNet-50 (BRECQ/ImageNet) |
| Input | 224x224x3 RGB (plain resize; ImageNet normalization compiled in the HEF) |
| Output | 1000 ImageNet-1k class scores |
| Parameters | 25.53M |
| Operations | 6.98G |
| Top-1 accuracy (float / Hailo-8) | 75.2% / 74.7% |
| HEF | Hailo Model Zoo v2.20.0, Hailo-8 (17,978,280 bytes) |
| License | MIT (BRECQ) |

## Quick Start

Runtime baseline: Python 3.11 and HailoRT 4.23. Run the build command from the
repository root.

```bash
docker build -t resnet_v1_50 -f docker/hailo8/resnet_v1_50.dockerfile src/rpi5_hailo8_resnet_v1_50

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23:/usr/lib/libhailort.so.4.23:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo8-cv/resnet_v1_50:latest \
  python web_detection.py --model_path model/resnet_v1_50.hef --video_path video/test.mp4
```

Open `http://<BOARD_IP>:8000` for the live preview. The default demo video is
Hailo's official `example.mp4` (downloaded from the `hailo-csdata` resource
bucket, MIT-licensed `hailo-apps` demo asset).

For a USB camera, add `--device /dev/video0:/dev/video0` and replace
`--video_path ...` with `--camera_id 0`.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web preview |
| `/api/video_feed` | GET | MJPEG stream |
| `/api/models/resnet_v1_50/predict` | POST | Top-K classification (JSON) |

```bash
curl -X POST "http://<BOARD_IP>:8000/api/models/resnet_v1_50/predict" \
  -F "file=@test.jpg" -F "conf=0.30" -F "iou=5"
```

`iou` is reused as the Top-K count for API parity with the other modules.

## Implementation notes

- The HEF exposes a single 1000-way score vector; `softmax` and `argmax` are
  disabled in the compile, so the host runs softmax and Top-K.
- Pre-processing is a plain resize to 224x224 and BGR->RGB; the ImageNet
  mean/std normalization is compiled into the HEF. No letterbox is used.
- The live preview smooths the probability vector (0.6 previous + 0.4 current)
  and draws the Top-K panel.
- The HEF input size is read at startup (224x224).

## Source

HEF from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)
(v2.20.0, Hailo-8):

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/resnet_v1_50.hef
```

Demo video: `https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/video/example.mp4`
(from `hailo-ai/hailo-apps`, MIT).

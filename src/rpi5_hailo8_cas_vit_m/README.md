# CAS-ViT-M - ImageNet Classification on CM5 + Hailo-8

CAS-ViT-M performs 1000-class ImageNet classification on **Hailo-8** through
HailoRT 4.23. The HEF takes a 384×384 BGR frame and returns 1000 scores; the
host applies softmax, smooths the probabilities over time and overlays a Top-K
panel on the MJPEG preview.

## Model

| Property | Value |
|---|---|
| Architecture | CAS-ViT-M (Convolutional Additive Self-attention Vision Transformer) |
| Input | 384×384×3 BGR (plain resize; bgr_to_rgb and ImageNet normalization compiled in the HEF) |
| Output | 1000 ImageNet-1k class scores |
| Parameters | 12.42M |
| Operations | 10.89G |
| Top-1 accuracy (float / Hailo-8) | 81.2% / 80.9% |
| Throughput (Hailo-8, batch 1) | 47.6 FPS |
| HEF | Hailo Model Zoo v2.20.0, Hailo-8 (18,544,130 bytes) |
| License | MIT ([Tianfang-Zhang/CAS-ViT](https://github.com/Tianfang-Zhang/CAS-ViT)) |

## Quick Start

Runtime baseline: Python 3.11 and HailoRT 4.23.0. Run the build command from the
repository root.

```bash
docker build -t cas_vit_m -f docker/hailo8/cas_vit_m.dockerfile src/rpi5_hailo8_cas_vit_m

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  cas_vit_m
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
| `/api/models/cas_vit_m/predict` | POST | Top-K classification (JSON) |

```bash
curl -X POST "http://<BOARD_IP>:8000/api/models/cas_vit_m/predict" \
  -F "file=@test.jpg" -F "conf=0.30" -F "iou=5"
```

Response:

```json
{
  "success": true,
  "source": "uploaded image",
  "predictions": [
    {"class": "sports car, sport car", "label": "sports car", "confidence": 0.87}
  ],
  "image": {"width": 1280, "height": 720}
}
```

`iou` is reused as the Top-K count for API parity with the other modules.

## Implementation notes

- The HEF exposes a single 1000-way score vector; the ALLS recipe contains no
  `softmax` or `argmax`, so the host runs softmax and Top-K. A probability
  heuristic also accepts a HEF that already emits normalized scores.
- Pre-processing is a plain resize to 384×384 and keeps **BGR**; the CAS-ViT
  ALLS recipe performs `bgr_to_rgb` and the ImageNet mean/std normalization
  inside the HEF. No letterbox is used and the app must not convert to RGB.
- The live preview smooths the probability vector (0.6 previous + 0.4 current)
  to stop the label flickering, then draws the Top-K panel.
- The Top-K count and confidence threshold are adjustable from the web UI
  (`/api/config`).

## Source

HEF from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) v2.20.0
(Hailo-8):

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/cas_vit_m.hef
```

Demo video: `https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/video/example.mp4`
(from `hailo-ai/hailo-apps`, MIT).
# MSPN RegNetX-800MF on Raspberry Pi 5 / CM5 + Hailo-8

This module runs MSPN (Multi-Stage Pose Network) with a RegNetX-800MF
backbone for single-person 2D pose estimation (17 COCO keypoints). The HEF
outputs a single heatmap (64×48×17); the app does argmax per channel to
extract keypoint coordinates, scales to input space, and draws the COCO
skeleton. The FastAPI service supports images, video files, USB cameras, an
MJPEG preview, and REST prediction endpoints.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-8 PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 4.23.x |
| Python | 3.11, aarch64 |
| Input | 256x192x3 RGB (normalize_in_net ImageNet RGB mean/std) |
| Output | Heatmap 64x48x17 (17 COCO keypoints) |
| Keypoints | 17 (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) |
| Parameters | 7.17M |
| Operations | 2.94G |
| HEF | Model Zoo v2.19.0, Hailo-8 |

## Build

```bash
sudo docker build -f docker/hailo8/mspn_regnetx_800mf.dockerfile \
  -t cm5-hailo8-mspn-regnetx-800mf:latest \
  src/rpi5_hailo8_mspn_regnetx_800mf
```

## Run

```bash
sudo docker run --rm --privileged --net=host \
  --name cm5-hailo8-mspn-regnetx-800mf \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  cm5-hailo8-mspn-regnetx-800mf:latest \
  python web_detection.py --model_path model/mspn_regnetx_800mf.hef --video_path video/test.mp4
```

USB camera mode adds `--device /dev/video0:/dev/video0` and runs with
`--camera_id 0`. Web preview: `http://<PI_IP>:8000`.

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/mspn_regnetx_800mf/predict" \
  -F "file=@test.jpg"
```

| Endpoint | Method | Description |
|---|---|---|
| `/api/models/mspn_regnetx_800mf/predict` | POST | 17 keypoints (JSON) |
| `/api/video_feed` | GET | MJPEG preview with skeleton overlay |

## Implementation notes

- **Single-person pose**: the model assumes the person is centered in the
  crop. For multi-person scenarios, a detector (YOLO) would need to crop each
  person first, then run MSPN per crop.
- **Post-processing**: argmax per heatmap channel → (x, y) in 64×48 space →
  scale to 256×192 input → un-letterbox to original frame. The Model Zoo
  reference implementation additionally applies a 5×5 Gaussian blur and a
  quarter-pixel shift; this demo skips both (sufficient for preview, matches
  the other pose module in this repo).
- **Output layout**: the raw vstream layout is logged once at first
  inference and verified on hardware. The parser accepts (1, 64, 48, 17) and
  (1, 17, 64, 48).
- `normalize_in_net` with ImageNet RGB mean/std; no input_conversion → feed
  raw uint8 RGB after letterboxing.

## Model source

[Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)
(model `mspn_regnetx_800mf`, v2.19.0).
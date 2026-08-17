# ViTPose-Small on Raspberry Pi 5 / CM5 + Hailo-8

This module runs ViTPose-Small single-person 2D pose estimation (17 COCO
keypoints). The HEF outputs a single heatmap (64×48×17); the app does argmax
per channel to extract keypoint coordinates, scales to input space, and draws
the COCO skeleton. The FastAPI service supports images, video files, USB
cameras, an MJPEG preview, and REST prediction endpoints.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-8 PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 4.23.x |
| Python | 3.11, aarch64 |
| Input | 256x192x3 RGB (normalize_in_net ImageNet RGB mean/std) |
| Output | Heatmap 64x48x17 (17 COCO keypoints) |
| Keypoints | 17 (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) |
| Parameters | 24.29M |
| Operations | 17.17G |
| HEF | Model Zoo v2.19.0, Hailo-8 |

## Build

```bash
sudo docker build -f docker/hailo8/vit_pose_small.dockerfile \
  -t vit_pose_small:latest \
  src/rpi5_hailo8_vit_pose_small
```

## Run

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  vit_pose_small:latest \
  python web_detection.py --model_path model/vit_pose_small.hef --video_path video/test.mp4
```

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/vit_pose_small/predict" \
  -F "file=@test.jpg"
```

| Endpoint | Method | Description |
|---|---|---|
| `/api/models/vit_pose_small/predict` | POST | 17 keypoints (JSON) |
| `/api/video_feed` | GET | MJPEG preview with skeleton overlay |

## Implementation notes

- **Single-person pose**: the model assumes the person is centered in the
  crop. For multi-person scenarios, a detector (YOLO) would need to crop each
  person first, then run ViTPose per crop.
- **Post-processing**: argmax per heatmap channel → (x, y) in 64×48 space →
  scale to 256×192 input → un-letterbox to original frame. No DARK sub-pixel
  refinement (sufficient for a demo; the official code has it).
- `normalize_in_net` with ImageNet RGB mean/std; no input_conversion → feed
  raw uint8 RGB after letterboxing.

## Model source

[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst)
(model `vit_pose_small`).

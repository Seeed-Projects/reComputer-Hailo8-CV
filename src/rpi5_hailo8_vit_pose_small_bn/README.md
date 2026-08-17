# ViTPose-Small-BN on Raspberry Pi 5 / CM5 + Hailo-8

Same architecture as ViTPose-Small but with BatchNorm variant (24.32M params,
AP 72.01). Identical I/O: 256×192 RGB input, 64×48×17 heatmap output, 17 COCO
keypoints. The FastAPI service supports images, video files, USB cameras, an
MJPEG preview, and REST prediction endpoints.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-8 PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 4.23.x |
| Python | 3.11, aarch64 |
| Input | 256x192x3 RGB |
| Output | Heatmap 64x48x17 (17 COCO keypoints) |
| Parameters | 24.32M |
| Operations | 17.17G |
| HEF | Model Zoo v2.19.0, Hailo-8 |

## Build

```bash
sudo docker build -f docker/hailo8/vit_pose_small_bn.dockerfile \
  -t vit_pose_small_bn:latest \
  src/rpi5_hailo8_vit_pose_small_bn
```

## Run

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  vit_pose_small_bn:latest \
  python web_detection.py --model_path model/vit_pose_small_bn.hef --video_path video/test.mp4
```

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/vit_pose_small_bn/predict" \
  -F "file=@test.jpg"
```

## Implementation notes

- Identical I/O and post-processing to vit_pose_small; only the BN variant
  differs (slightly lower AP: 72.01 vs 74.16).
- Single-person pose: argmax per heatmap channel → keypoint coords → scale →
  un-letterbox → draw COCO 17-keypoint skeleton.

## Model source

[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst)
(model `vit_pose_small_bn`).

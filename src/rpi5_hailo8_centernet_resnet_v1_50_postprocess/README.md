# CenterNet (resnet_v1_50) on Raspberry Pi 5 / CM5 + Hailo-8

This module runs CenterNet object detection (ResNet-50 backbone, COCO 80
classes) with the on-chip `max_finder` post-processing variant. It is the
higher-accuracy sibling of the resnet_v1_18 build: same input/output
contract, deeper backbone. The FastAPI service supports images, video
files, USB cameras, an MJPEG preview, and REST prediction endpoints.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-8 PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 4.23.x |
| Python | 3.11, aarch64 |
| Input | 512x512x3 RGB (normalization compiled into the HEF) |
| Output | 3 heads: wh (128x128x2), reg (128x128x2), sparse heatmap (128x128x80) |
| Classes | 80 (COCO) |
| Parameters | 30.07M |
| Operations | 56.92G |
| HEF | Model Zoo v2.19.0, Hailo-8 |

The host driver, firmware, `libhailort.so`, and Python wheel must use the same
HailoRT major/minor version.

## Build

From the repository root:

```bash
sudo docker build -f docker/hailo8/centernet_resnet_v1_50_postprocess.dockerfile \
  -t centernet_resnet_v1_50_postprocess:latest \
  src/rpi5_hailo8_centernet_resnet_v1_50_postprocess
```

## Run the demo video

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  centernet_resnet_v1_50_postprocess:latest \
  python web_detection.py --model_path model/centernet_resnet_v1_50_postprocess.hef --video_path video/test.mp4
```

Open `http://<PI_IP>:8000`. For a USB camera, mount `/dev/video0` and replace
`--video_path video/test.mp4` with `--camera_id 0`.

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/centernet_resnet_v1_50_postprocess/predict" \
  -F "file=@test.jpg"
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/api/models/centernet_resnet_v1_50_postprocess/predict` | POST | Detections (JSON) |
| `/api/video_feed` | GET | MJPEG preview stream |

## Implementation notes

- Identical input/output contract and post-processing to the resnet_v1_18
  build; only the backbone differs (ResNet-50 → more accurate, heavier).
- The HEF uses `normalize_in_net=true` (ImageNet mean/std, RGB); the app feeds
  raw uint8 RGB pixels after letterboxing — no manual normalization.
- The heatmap head is already sparse: the on-chip `max_finder` + score
  threshold (0.2) keep only local maxima above threshold. The decoder takes
  those peaks, reads `wh`/`reg` at each peak, and reconstructs boxes with the
  official Hailo Model Zoo formula (stride 4, 128x128 feature map -> 512x512
  input).
- First inference prints every output vstream name/shape and the resolved
  hm/wh/reg mapping, so the head assignment can be verified on hardware.

## Model source

The Hailo-8 HEF comes from the
[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst).

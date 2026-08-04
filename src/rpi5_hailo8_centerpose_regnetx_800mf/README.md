# CenterPose RegNetX-800MF on Raspberry Pi 5 + Hailo-8

This module provides multi-person COCO pose estimation with CenterPose and a
RegNetX-800MF backbone. It reuses the `yolov8_pose` video/web service template
from `reComputer-R20-CV`, while using CenterPose-specific preprocessing and
CenterNet-style decoding for the six HEF output heads.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-8 PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 4.23.x |
| Python | 3.11, aarch64 |
| Input | 512×512×3 BGR |
| Output | Person boxes and 17 COCO keypoints |
| Parameters | 12.31M |
| HEF | Model Zoo v2.19.0, Hailo-8 |

The host driver, firmware, `libhailort.so`, and Python wheel must use the same
HailoRT major/minor version.

## Build

From the repository root:

```bash
sudo docker build -f docker/hailo8/centerpose_regnetx_800mf.dockerfile \
  -t centerpose_regnetx_800mf:latest \
  src/rpi5_hailo8_centerpose_regnetx_800mf
```

## Run the demo video

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  centerpose_regnetx_800mf:latest \
  python web_detection.py \
    --model_path model/centerpose_regnetx_800mf.hef \
    --video_path video/test.mp4
```

Open `http://<PI_IP>:8000`.

For a USB camera, mount `/dev/video0` and replace `--video_path ...` with
`--camera_id 0`.

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/centerpose_regnetx_800mf/predict" \
  -F "file=@test.jpg"
```

The response contains person boxes, confidence values, and visible COCO
keypoints. The service also exposes MJPEG preview and offline video analysis.

## Implementation notes

- `web_detection.py` maps v2.19 output layers `conv60`–`conv65` to `hm`, `wh`,
  `hps`, `reg`, `hm_hp`, and `hp_offset`.
- The HEF contains BGR normalization, so preprocessing deliberately does not
  swap OpenCV frames to RGB.
- The first inference logs every vstream name and shape for deployment checks.


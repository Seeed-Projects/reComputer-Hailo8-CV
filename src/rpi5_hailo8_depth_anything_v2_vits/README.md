# Depth-Anything-V2-ViTS on Raspberry Pi 5 / CM5 + Hailo-8

This module runs Depth-Anything-V2-Small (ViT-S backbone) for zero-shot
monocular depth estimation. The HEF outputs a 224×224×1 relative depth map;
the app normalizes it, applies an INFERNO colormap, and alpha-blends onto the
original frame. The FastAPI service supports images, video files, USB cameras,
an MJPEG preview, and REST prediction endpoints.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-8 PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 4.23.x |
| Python | 3.11, aarch64 |
| Input | 224x224x3 RGB (normalize_in_net ImageNet RGB mean/std) |
| Output | Depth map 224x224x1 (relative, zero-shot) |
| Parameters | 24.2M |
| Operations | 16.7G |
| AbsRel | 0.147 |
| HEF | Model Zoo v2.19.0, Hailo-8 |

## Build

```bash
sudo docker build -f docker/hailo8/depth_anything_v2_vits.dockerfile \
  -t depth_anything_v2_vits:latest \
  src/rpi5_hailo8_depth_anything_v2_vits
```

## Run

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  depth_anything_v2_vits:latest \
  python web_detection.py --model_path model/depth_anything_v2_vits.hef --video_path video/test.mp4
```

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/depth_anything_v2_vits/predict" \
  -F "file=@test.jpg"
```

Response returns depth statistics (min, max, mean, std).

## Implementation notes

- **Zero-shot relative depth**: the output is relative depth (not metric).
  Higher values = farther. The app min-max normalizes per-frame for
  visualization.
- **Visualization**: INFERNO colormap (dark=close, bright=far) alpha-blended
  onto the original frame. The slider controls the blend alpha.
- `normalize_in_net` with ImageNet RGB mean/std; no input_conversion → feed
  raw uint8 RGB after letterboxing.
- No on-chip post-processing — the depth map is the raw HEF output (FLOAT32).

## Model source

[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst)
(model `depth_anything_v2_vits`, source: [Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)).

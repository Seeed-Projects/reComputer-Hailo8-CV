# Depth-Anything-ViTS on Raspberry Pi 5 / CM5 + Hailo-8

Same architecture as Depth-Anything-V2-ViTS but the V1 version (AbsRel 0.13,
slightly more accurate). Identical I/O: 224×224 RGB → 224×224×1 relative depth
map, INFERNO colormap visualization, depth statistics API.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-8 PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 4.23.x |
| Python | 3.11, aarch64 |
| Input | 224x224x3 RGB (normalize_in_net ImageNet RGB) |
| Output | Depth map 224x224x1 (relative, zero-shot) |
| Parameters | 24.2M |
| Operations | 16.7G |
| AbsRel | 0.13 |
| HEF | Model Zoo v2.19.0, Hailo-8 |

## Build

```bash
sudo docker build -f docker/hailo8/depth_anything_vits.dockerfile \
  -t depth_anything_vits:latest \
  src/rpi5_hailo8_depth_anything_vits
```

## Run

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  depth_anything_vits:latest \
  python web_detection.py --model_path model/depth_anything_vits.hef --video_path video/test.mp4
```

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/depth_anything_vits/predict" \
  -F "file=@test.jpg"
```

Returns depth statistics (min, max, mean, std).

## Implementation notes

- Identical I/O and post-processing to V2-ViTS; V1 is slightly more accurate
  (AbsRel 0.13 vs 0.147).
- Zero-shot relative depth, INFERNO colormap, alpha-blend overlay.
- `normalize_in_net` ImageNet RGB; feed raw uint8 RGB.

## Model source

[Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection.rst)
(model `depth_anything_vits`).

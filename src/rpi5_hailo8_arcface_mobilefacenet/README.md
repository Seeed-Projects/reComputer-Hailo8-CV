# ArcFace MobileFaceNet on Raspberry Pi 5 + Hailo-8

This module extracts 512-dim face embeddings with ArcFace + MobileFaceNet on a
Hailo-8 accelerator. It reuses the CenterPose application template (FastAPI
service, MJPEG preview, offline video analysis, USB camera support) with
ArcFace-specific preprocessing and a single-output embedding post-process.

Face comparison is performed by the client: the service returns the embedding
for an uploaded image and does not keep a face database.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-8 PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 4.23.x |
| Python | 3.11, aarch64 |
| Input | 112×112×3 RGB (letterboxed, uint8) |
| Output | 512-dim face embedding |
| Parameters | 2.04M (Hailo Model Zoo) |
| HEF | Model Zoo v2.19.0, Hailo-8 |

The host driver, firmware, `libhailort.so`, and Python wheel must use the same
HailoRT major/minor version.

## Build

From the repository root:

```bash
sudo docker build -f docker/hailo8/arcface_mobilefacenet.dockerfile \
  -t arcface_mobilefacenet:latest \
  src/rpi5_hailo8_arcface_mobilefacenet
```

## Run the demo video

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  arcface_mobilefacenet:latest \
  python web_detection.py \
    --model_path model/arcface_mobilefacenet.hef \
    --video_path video/test.mp4
```

Open `http://<PI_IP>:8000`.

For a USB camera, mount `/dev/video0` and replace `--video_path ...` with
`--camera_id 0`.

## REST API

```bash
curl -X POST "http://<PI_IP>:8000/api/models/arcface_mobilefacenet/predict" \
  -F "file=@face.jpg"
```

```json
{
  "success": true,
  "source": "uploaded image",
  "embedding": [0.023, -0.156, 0.089, "..."],
  "dimension": 512,
  "image": { "width": 1280, "height": 720 }
}
```

The service also exposes MJPEG preview, uploaded-video analysis, and camera
mode through the same web UI.

## Implementation notes

- `preprocess_frame()` letterboxes the frame to 112×112 and converts BGR to
  RGB; the HEF carries its own normalization, so no host-side scaling is done.
- `post_process_hailo()` selects the embedding vstream (the `fc1` output when
  the name is exposed) and returns it as a 512-element float vector.
- The first inference logs every vstream name and shape so the output mapping
  can be verified on hardware.
- `--class_path` is accepted for CLI compatibility but ignored: ArcFace has no
  class label table.
- The embedding is L2-normalized on the client side when cosine similarity is
  used for verification.

## Troubleshooting

| Symptom | Check |
|---|---|
| `HailoRT is not available` | Install the wheel in `hailort-packages/` matching the host HailoRT version |
| Firmware version error | Host driver, `libhailort.so`, and container wheel must all be 4.23.x |
| Embedding looks constant | Confirm the vstream logged on first inference is `fc1` and that inputs are RGB |
| Low similarity for the same face | Letterboxed faces are small; crop the face before sending it |
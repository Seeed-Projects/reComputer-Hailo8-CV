# ByteTrack-S (MOT17) on CM5 + Hailo-8

This module runs the ByteTrack detector for MOT17-style pedestrian tracking on
**Hailo-8** through HailoRT 4.23.0. The HEF exposes nine raw YOLOX heads
(box, objectness and class per stride); the host decodes boxes, runs NMS and
feeds them into a ByteTrack association stage. The MJPEG preview draws a stable
track ID, class and score per person.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-8 PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 4.23.0 |
| Python | 3.11, aarch64 |
| Input | 608x1088x3 RGB (letterbox, padding 114; ImageNet normalization in HEF) |
| Output | 9 raw heads: 19x34 / 38x68 / 76x136, each box(4) + obj(1) + class(1) |
| Classes | 1 (person, MOT17) |
| Parameters | 8.94M |
| Operations | 43.21G |
| MOTA (float / Hailo-8) | 79.5 / 77.7 |
| FPS (Hailo-8, batch 1) | 88.3 |
| HEF | Model Zoo v2.20.0, Hailo-8 (14,836,099 bytes) |

## Build

From the repository root:

```bash
sudo docker build -f docker/hailo8/bytetrack_s_mot17.dockerfile \
  -t bytetrack_s_mot17:latest \
  src/rpi5_hailo8_bytetrack_s_mot17
```

## Run the demo video

```bash
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  bytetrack_s_mot17:latest \
  python web_detection.py --model_path model/bytetrack_s_mot17.hef --video_path video/test.mp4
```

Open `http://<BOARD_IP>:8000` for the track-ID overlay. The bundled
`video/test.mp4` is the ByteTrack palace pedestrian demo clip from the upstream
ByteTrack repository. For a USB camera, mount `/dev/video0` and
replace `--video_path ...` with `--camera_id 0`.

## REST API

```bash
curl -X POST "http://<BOARD_IP>:8000/api/models/bytetrack_s_mot17/predict" \
  -F "file=@test.jpg"
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/api/video_feed` | GET | MJPEG preview with track IDs |
| `/api/models/bytetrack_s_mot17/predict` | POST | Person detections with track IDs |

Each prediction contains `track_id`, `class`, `confidence` and `box`.

## Implementation notes

- HEF outputs nine YOLOX-style branches (box/obj/class for strides 32, 16, 8);
  there is no on-chip NMS.
- Host decode: `center = (raw_xy + grid) * stride`,
  `size = exp(raw_wh) * stride`, `score = obj * class` (the HEF already applies
  sigmoid). Boxes are filtered at 0.1 and NMS is 0.7.
- Boxes are associated by ByteTrack (Kalman prediction + high/low-score IoU
  matching) in `py_utils/byte_tracker.py`; only activated tracks are drawn.
- Pre-processing letterboxes to 608x1088 with padding 114 and converts BGR to
  RGB; ImageNet normalization is compiled into the HEF.
- This is a person-only MOT17 detector; it is not a generic 80-class detector.

## Model source

The Hailo-8 HEF comes from
[Hailo Model Zoo v2.20.0](https://github.com/hailo-ai/hailo_model_zoo) (model
`bytetrack_s_mot17`).

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/bytetrack_s_mot17.hef
```

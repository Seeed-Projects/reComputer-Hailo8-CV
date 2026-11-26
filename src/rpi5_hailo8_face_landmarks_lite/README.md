# Face Landmarks Lite on Raspberry Pi 5 + Hailo-8

MediaPipe Face Landmarks Lite (0.6M parameters) on Hailo-8, with a YuNet CPU
face detector supplying the face ROI. The Hailo HEF returns **468 facial
landmarks per face**, not 98 points.

> The Hailo Model Zoo publishes the landmark HEF only. Its official
> `face_detection_landmarks` multi-network pairs `lightface_slim` with
> `tddfa_mobilenet_v1`, not Face Landmarks Lite. This module therefore uses
> YuNet on the CPU for face detection and the Hailo-8 HEF for landmarks. The
> CPU detector is explicit in the API and documentation.

## Features

- 468-point facial landmark inference on Hailo-8.
- YuNet face detection and square ROI cropping on the CPU.
- MJPEG preview with face boxes and landmarks.
- REST prediction for images, video frames, and the live camera frame.
- Offline video analysis with libx264 encoding when `ffmpeg` is available.
- Configurable face confidence, NMS IoU, and maximum faces.

## Model

| Property | Value |
|---|---|
| Architecture | MediaPipe Face Landmarks Lite |
| Hailo HEF | v2.20.0, `hailo8/face_landmarks_lite.hef` |
| HEF size | 3,135,094 bytes |
| Input | 192x192x3 RGB face crop |
| Output | 1404 landmark values + 1 face-presence value |
| Landmarks | 468 x (x, y, z) |
| Parameters | 0.6M |
| Operations | 0.07G |
| Official FPS | 601 (batch 1) / 2317 (batch 8) on Hailo-8 |
| Full-precision NME | 4.28 |
| Face detector | OpenCV YuNet 2023mar, CPU, MIT |
| License | Apache-2.0 (MediaPipe model) |

## Runtime baseline

- Python 3.11, aarch64
- HailoRT 4.23.0
- Host driver, firmware, `libhailort.so`, and the Python wheel must share the
  same major.minor version
- `opencv-python` with `FaceDetectorYN` support

## Host setup

```bash
sudo apt update
sudo apt install hailo-all -y
sudo reboot

hailortcli fw-control identify
ls /dev/hailo0
```

## Build and run

```bash
cd src/rpi5_hailo8_face_landmarks_lite
sudo docker build -f ../../docker/hailo8/face_landmarks_lite.dockerfile \
  -t rpi5-hailo8-face-landmarks:latest .

sudo docker run --rm --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  rpi5-hailo8-face-landmarks:latest
```

Open `http://<Board_IP>:8000` for the MJPEG preview.

Camera mode:

```bash
sudo docker run --rm --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  --device /dev/video0:/dev/video0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  rpi5-hailo8-face-landmarks:latest \
  python web_detection.py --model_path model/face_landmarks_lite.hef --camera_id 0
```

## REST API

```bash
curl -X POST "http://<Board_IP>:8000/api/models/face_landmarks_lite/predict" \
  -F "file=@face.jpg"
```

The response contains one entry per detected face. Each entry has a face
`box`, a `score`, and `landmarks` with 468 `{x, y, z}` points mapped back to
the original image. `conf` and `iou` are optional form fields.

## Implementation notes

- YuNet detects faces on a frame resized to at most 960 pixels on the long
  side; boxes are scaled back to the original frame.
- Each face box is expanded by 20%, made square, cropped, resized to
  192x192, and converted from BGR to RGB before Hailo inference.
- The HEF returns 1404 landmark values. The decoder accepts both normalized
  `[0, 1]` coordinates and 192-pixel coordinates; the scale is selected from
  the observed x/y range and logged on the first frame.
- Landmark coordinates are mapped from the 192x192 crop back to the original
  frame. `z` is passed through as the model's relative depth.
- The HEF's second output is a face-presence value; it is converted with a
  sigmoid if it is outside `[0, 1]` and reported as `landmark_score`.

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | MJPEG preview |
| `/api/video_feed` | GET | MJPEG stream |
| `/api/config` | GET/POST | Face confidence, NMS IoU, maximum faces |
| `/api/models/face_landmarks_lite/predict` | POST | 468 landmarks per face |
| `/api/video/upload` | POST | Upload a video for offline analysis |
| `/api/video/list` | GET | List uploaded videos |
| `/api/video/analyze` | POST | Start offline analysis |
| `/api/video/status` | GET | Analysis progress |
| `/api/video/download/{filename}` | GET | Download annotated video |

## Development notes

- Source: `src/rpi5_hailo8_face_landmarks_lite/`
- Dockerfile: `docker/hailo8/face_landmarks_lite.dockerfile`
- HEF: Hailo Model Zoo v2.20.0 `hailo8`; URL and SHA256 in `model/README.md`
- Face detector: OpenCV Zoo YuNet 2023mar; MIT license in `model/YUNET_LICENSE.txt`
- Tests: `python -m unittest discover -s tests -v`
- Hardware validation is still pending; see `TEST_REPORT.md`.
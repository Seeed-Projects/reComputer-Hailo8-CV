# YOLO26s on CM5 + Hailo-8

YOLO26s performs COCO 80-class object detection on **Hailo-8** through HailoRT.
It is a one2one head (one prediction per grid cell, no anchors, no NMS), so the
application decodes the raw split heads and selects with ultralytics' two-stage
top-k; a compile that ships the on-chip HPP NMS result is parsed as well. The
FastAPI service provides image prediction, video and camera input, an MJPEG
preview and offline video analysis.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-8 PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 4.23.x |
| Python | 3.11, aarch64 |
| Input | 640x640x3 RGB (normalize_in_net mean 0 / std 255, padding 114) |
| Output | raw split heads **or** on-chip NMS (logged on the first inference) |
| Classes | 80 (COCO, 0-indexed) |
| Parameters | 9.5M |
| Operations | 20.9G |
| HEF | Hailo Model Zoo v2.19.0 |
| Model licence | AGPL-3.0 (upstream: ultralytics/ultralytics) |

The host driver, firmware, `libhailort.so`, and Python wheel must use the same
HailoRT major/minor version.

## Build

From the repository root:

```bash
sudo docker build -f docker/hailo8/yolo26s.dockerfile \
  -t yolo26s:latest \
  src/rpi5_hailo8_yolo26s
```

## Run the demo video

```bash
sudo docker run --rm \
  --name cm5-hailo8-yolo26s \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo8-cv/yolo26s:latest \
  python web_detection.py --model_path model/yolo26s.hef --video_path video/test.mp4
```

Open `http://<BOARD_IP>:8000`.

For a USB camera, mount `/dev/video0` and replace `--video_path ...` with
`--camera_id 0`.

## REST API

```bash
curl -X POST "http://<BOARD_IP>:8000/api/models/yolo26s/predict" \
  -F "file=@bus.jpg" -F "conf=0.25"
```

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Web preview UI |
| `/api/models/yolo26s/predict` | POST | Detections (JSON) |
| `/api/video_feed` | GET | MJPEG preview stream |
| `/api/config` | GET / POST | Read or change runtime thresholds |
| `/api/video/upload` | POST | Upload a source video |
| `/api/video/analyze` | POST | Start offline video analysis |
| `/api/video/status` | GET | Read analysis progress |
| `/api/video/download/{filename}` | GET | Download the annotated result |

## Implementation notes

- Raw-head layout: per stride 32/16/8 a 4-channel box tensor (distances from
  the cell centre to l, t / r, b in stride units; `regression_length=1`) and an
  80-channel class tensor. The host applies sigmoid, the two-stage top-k
  (`post_nms_topk=300`, no NMS) and clips the boxes to the input.
- On-chip NMS layout: the post-NMS tensor is parsed instead (compact per-class
  buffer, dense `Cx5xD` / `CxDx5`, ragged NMS-by-score list); no host NMS runs.
- Pre-processing letterboxes to 640x640x3 with gray (114) padding, converts BGR
  to RGB and feeds raw uint8; the `/255` normalization is compiled into the HEF.
- `cls_id` (0..79) indexes the standard COCO class list directly.
- The first inference logs the layout plus the box/score ranges
  (`[YOLO26] layout=..., outputs=[...]`).

## Hardware acceptance checklist

1. `hailortcli --version` reports 4.23.x and `/dev/hailo0` exists.
2. The startup log reports the HEF input size and the first inference reports
   the output layout.
3. The demo video shows COCO boxes with class labels.
4. `POST /api/models/yolo26s/predict` returns detections with class,
   confidence and box.
5. USB camera mode keeps the preview live without stale frames.

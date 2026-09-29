# YOLOv6n on CM5 + Hailo-8

YOLOv6n performs COCO 80-class object detection on **Hailo-8** through HailoRT.
The compiled HEF either returns the on-chip HPP NMS result or the nine raw
split heads; the application handles both (Model Zoo `base/yolov6.yaml`:
`hpp=true`, `meta_arch=yolo_v6`, `score_threshold=0.03`,
`nms_iou_thresh=0.65`). The FastAPI service provides image prediction, video
and camera input, an MJPEG preview and offline video analysis.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-8 PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 4.23.0 |
| Python | 3.11, aarch64 |
| Input | 640x640x3 RGB (normalize_in_net mean 0 / std 255, padding 114) |
| Output | on-chip NMS tensor **or** nine raw heads (confirmed by the first-inference log) |
| Classes | 80 (COCO, 0-indexed) |
| Parameters | 4.32M |
| Operations | 11.12G |
| HEF | Hailo Model Zoo v5.4.0 (Hailo-10H) / v2.19.0 (Hailo-8) |
| Model licence | GPL-3.0 (upstream: meituan/YOLOv6) |

The host driver, firmware, `libhailort.so`, and Python wheel must use the same
HailoRT major/minor version.

## Build

From the repository root:

```bash
sudo docker build -f docker/hailo8/yolov6n.dockerfile \
  -t yolov6n:latest \
  src/rpi5_hailo8_yolov6n
```

## Run the demo video

```bash
sudo docker run --rm \
  --name cm5-hailo8-yolov6n \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo8-cv/yolov6n:latest \
  python web_detection.py --model_path model/yolov6n.hef --video_path video/test.mp4
```

Open `http://<BOARD_IP>:8000`.

For a USB camera, mount `/dev/video0` and replace `--video_path ...` with
`--camera_id 0`.

## REST API

```bash
curl -X POST "http://<BOARD_IP>:8000/api/models/yolov6n/predict" \
  -F "file=@bus.jpg" -F "conf=0.25" -F "iou=0.65"
```

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Web preview UI |
| `/api/models/yolov6n/predict` | POST | Detections (JSON) |
| `/api/video_feed` | GET | MJPEG preview stream |
| `/api/config` | GET / POST | Read or change runtime thresholds |
| `/api/video/upload` | POST | Upload a source video |
| `/api/video/analyze` | POST | Start offline video analysis |
| `/api/video/status` | GET | Read analysis progress |
| `/api/video/download/{filename}` | GET | Download the annotated result |

## Implementation notes

- Layout (a) — the HEF already ran NMS (HPP): the app parses the post-NMS
  tensor (compact per-class buffer, dense `Cx5xD` / `CxDx5`, ragged
  NMS-by-score list) and `nms_thresh` is accepted for API parity only.
- Layout (b) — the HEF returns the nine raw split heads (4 box, 1 objectness,
  80 classes per stride 32/16/8): the app decodes the box distances around the
  cell centre (stride units), multiplies sigmoid(classes) by sigmoid(objectness)
  and runs a per-class NMS with the IOU slider value (Model Zoo default 0.65).
- Pre-processing letterboxes to 640x640x3 with gray (114) padding, converts BGR
  to RGB and feeds raw uint8; the `/255` normalization is compiled into the HEF.
- `cls_id` (0..79) indexes the standard COCO class list directly; the Model Zoo
  evaluation uses `labels_offset=1` for COCO category ids.
- The first inference logs which layout the HEF returned
  (`[YOLOv6n] layout=..., outputs=[...]`) so it can be confirmed on hardware.

## Hardware acceptance checklist

1. `hailortcli --version` reports 4.23.0 and `/dev/hailo0` exists.
2. The startup log reports the HEF input size (`Model input size: ...`) and the
   first inference reports the output layout.
3. The demo video shows COCO boxes with class labels.
4. `POST /api/models/yolov6n/predict` returns detections with class, confidence
   and box.
5. USB camera mode keeps the preview live without stale frames.

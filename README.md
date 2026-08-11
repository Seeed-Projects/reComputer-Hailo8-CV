# reComputer-Hailo8-CV

[English] | [中文](./README_zh.md)

Industrial-grade computer-vision reference for **Raspberry Pi 5 / CM5 + Hailo-8**
(reComputer R Series). Each model ships as an independent module with a
FastAPI service: real-time MJPEG preview, REST prediction, USB camera, and
offline batch video analysis — built around the PCIe-attached Hailo-8
accelerator and HailoRT 4.23.x.

The repo covers three task families — **object detection** (CenterNet,
DAMO-YOLO), **semantic segmentation** (STDC1), and **pose estimation**
(CenterPose). Every module follows the same skeleton (HailoRT executor,
letterbox + coordinate restore, frame buffer, MJPEG encode); only the
preprocessing, on-device post-processing mapping, and decode differ per HEF.
Treat any module as a template and retarget to other Hailo Model Zoo models,
but always re-derive the post-processing from the real HEF output — never
just swap the file name.

---

## Hardware platform

| | |
|---|---|
| Board | Raspberry Pi 5 / CM5 (reComputer R Series carrier) |
| Accelerator | Hailo-8 M.2 (PCIe), device node `/dev/hailo0` |
| OS | Raspberry Pi OS Bookworm, kernel 6.12+ aarch64 |
| Host drivers | `hailo-all` apt package (PCIe driver, firmware, `libhailort.so`) |
| HailoRT | 4.23.x validated — host driver / firmware / container wheel **must share major.minor** |

---

## Included models

| Model | Task | Parameters | Module | Container image |
|---|---|---:|---|---|
| CenterPose RegNetX-800MF | Pose (17 COCO keypoints) | 12.31M | `src/rpi5_hailo8_centerpose_regnetx_800mf/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/centerpose_regnetx_800mf:latest` |
| STDC1 | Semantic segmentation (Cityscapes 19) | 8.27M | `src/rpi5_hailo8_stdc1/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/stdc1:latest` |
| CenterNet (resnet_v1_18) | Object detection (COCO 80) | 14.22M | `src/rpi5_hailo8_centernet_resnet_v1_18_postprocess/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/centernet_resnet_v1_18_postprocess:latest` |
| CenterNet (resnet_v1_50) | Object detection (COCO 80) | 30.07M | `src/rpi5_hailo8_centernet_resnet_v1_50_postprocess/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/centernet_resnet_v1_50_postprocess:latest` |
| DAMO-YOLO (tinynasL20_T) | Object detection (COCO 80) | 11.35M | `src/rpi5_hailo8_damoyolo_tinynas_l20_t/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/damoyolo_tinynas_l20_t:latest` |
| DAMO-YOLO (tinynasL25_S) | Object detection (COCO 80) | 16.25M | `src/rpi5_hailo8_damoyolo_tinynas_l25_s/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/damoyolo_tinynas_l25_s:latest` |

All HEFs come from [Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo) (Hailo-8 target).

---

## Quick start (pre-built image)

The example below uses the published DAMO-YOLO (L25_S) image, which already
bundles the source, HailoRT wheel, ffmpeg, the `.hef`, and a demo video. You
only need a working Hailo toolchain on the host.

### 1. Host prep (one-time, on the Pi)

#### Install Docker

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh --mirror Aliyun
sudo systemctl enable docker
sudo systemctl start docker
```

#### Install Hailo toolchain

```bash
sudo apt update
sudo apt install hailort hailort-pcie-driver python3-hailort
sudo reboot

# After reboot, confirm the chip and note the firmware version
hailortcli fw-control identify     # should report 4.23.x
ls /dev/hailo0
```

> Install the three HailoRT packages directly — they come from the Raspberry Pi
> OS archive at 4.23.x and cover runtime, PCIe driver + firmware, and the
> Python API. Avoid the `hailo-all` meta-package: it can pull newer Hailo-10H /
> 5.x packages that don't match the Hailo-8 4.23.x baseline.

### 2. Run

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    ghcr.io/seeed-projects/recomputer-hailo8-cv/damoyolo_tinynas_l25_s:latest
```

Docker pulls the image on first run. The container then loops the bundled
`video/test.mp4` and serves the Web UI on port `8000` — open
`http://<device_IP>:8000` in a browser.

> **Why the `libhailort.so` bind-mount?** The image ships only the Python
> bindings; the native library must come from the host's `hailo-all` package.
> If your firmware version isn't `4.23.0`, replace both `4.23.0` references
> with the version printed by `hailortcli fw-control identify` (and rebuild
> from source against a matching wheel if the major.minor differs).

### USB camera mode

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    --device /dev/video0:/dev/video0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    ghcr.io/seeed-projects/recomputer-hailo8-cv/damoyolo_tinynas_l25_s:latest \
    python web_detection.py --model_path model/damoyolo_tinynas_l25_s.hef --camera_id 0
```

---

## Repository layout

```text
reComputer-Hailo8-CV/
├── .github/workflows/build-ghcr-images.yml   # Per-model GHCR build (only changed models rebuild)
├── docker/hailo8/
│   ├── centerpose_regnetx_800mf.dockerfile
│   ├── stdc1.dockerfile
│   ├── centernet_resnet_v1_18_postprocess.dockerfile
│   ├── centernet_resnet_v1_50_postprocess.dockerfile
│   ├── damoyolo_tinynas_l20_t.dockerfile
│   └── damoyolo_tinynas_l25_s.dockerfile
└── src/
    ├── rpi5_hailo8_centerpose_regnetx_800mf/
    ├── rpi5_hailo8_stdc1/
    ├── rpi5_hailo8_centernet_resnet_v1_18_postprocess/
    ├── rpi5_hailo8_centernet_resnet_v1_50_postprocess/
    ├── rpi5_hailo8_damoyolo_tinynas_l20_t/
    └── rpi5_hailo8_damoyolo_tinynas_l25_s/

# Per-module layout (same skeleton for all):
src/rpi5_hailo8_<slug>/
    ├── web_detection.py            # FastAPI + inference/encode threading pipeline
    ├── py_utils/
    │   ├── hailo_executor.py        # HailoRT wrapper, long-lived InferVStreams
    │   └── coco_utils.py           # Letterbox + box/mask coordinate restore
    ├── model/<slug>.hef             # Hailo-8 HEF (bundled)
    ├── hailort-packages/            # HailoRT wheel (bundled)
    ├── video/test.mp4               # Bundled demo source
    ├── requirements.txt
    ├── README.md / README_zh.md     # Module deep dive: build, CLI, troubleshooting
    └── TEST_REPORT.md               # Validation log
```

---

## Build from source

For customization — swapping a `.hef`, changing the code, or rebuilding against
a different HailoRT version:

```bash
git clone https://github.com/Seeed-Projects/reComputer-Hailo8-CV.git
cd reComputer-Hailo8-CV/src/rpi5_hailo8_damoyolo_tinynas_l25_s

sudo docker build -f ../../docker/hailo8/damoyolo_tinynas_l25_s.dockerfile \
    -t hailo8-damoyolo-l25s:latest .

# Same run command, with the local tag instead of ghcr.io
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    hailo8-damoyolo-l25s:latest
```

---

## REST API

All endpoints listen on port `8000` of the container; with `--net=host` they're
reachable at `http://<device_IP>:8000`. Replace `<slug>` with the model slug
from the table above (e.g. `damoyolo_tinynas_l25_s`).

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/models/<slug>/predict` | POST | One-shot inference on uploaded image, specific video frame, or current camera frame |
| `/api/video_feed` | GET | MJPEG live stream with results overlaid (embed in an `<img>`) |
| `/api/config` | GET / POST | Read or update `obj_thresh` / `nms_thresh` |
| `/api/video/upload` | POST | Upload a video for batch analysis |
| `/api/video/analyze` | POST | Start an offline analysis job (form-data `filename=...`) |
| `/api/video/status` | GET | Poll job progress |
| `/api/video/list` | GET | List uploaded sources and finished outputs |
| `/api/video/download/{filename}` | GET | Download an annotated output |

### Inference examples

```bash
# Image upload
curl -X POST http://<device_IP>:8000/api/models/damoyolo_tinynas_l25_s/predict -F "file=@test.jpg"

# Specific frame of an uploaded video (timestamp in seconds)
curl -X POST http://<device_IP>:8000/api/models/damoyolo_tinynas_l25_s/predict \
    -F "video=@test.mp4" -F "timestamp=5.5"

# Current camera frame
curl -X POST http://<device_IP>:8000/api/models/damoyolo_tinynas_l25_s/predict -F "realtime=true"

# Per-call threshold override
curl -X POST http://<device_IP>:8000/api/models/damoyolo_tinynas_l25_s/predict \
    -F "file=@test.jpg" -F "conf=0.5" -F "iou=0.4"
```

Detection response:

```json
{
  "success": true,
  "source": "uploaded image",
  "predictions": [
    {
      "class": "car",
      "confidence": 0.91,
      "box": { "x1": 100, "y1": 120, "x2": 320, "y2": 520 }
    }
  ],
  "image": { "width": 1280, "height": 720 }
}
```

Embed the live stream in any HTML page:

```html
<img src="http://<device_IP>:8000/api/video_feed">
```

### Dynamic threshold update

```bash
# Read current
curl http://<device_IP>:8000/api/config
# {"obj_thresh":0.25,"nms_thresh":0.45}

# Update (either field is optional)
curl -X POST http://<device_IP>:8000/api/config \
     -H "Content-Type: application/json" \
     -d '{"obj_thresh":0.4}'
```

> Some HEFs perform NMS / peak-finding on-chip (CenterNet `max_finder`), others
> only apply sigmoid on-chip and decode on CPU (DAMO-YOLO `nanodet_split`).
> The thresholds always filter the final output regardless of where NMS runs.

---

## Adapting to other models

Treat any module under `src/` as a template:

1. Copy the directory and rename (e.g. `rpi5_hailo8_<new_slug>/`).
2. Drop the new `.hef` into `model/`, renamed to `<slug>.hef` (lowercase, for
   GHCR image-name validity).
3. Add `docker/hailo8/<slug>.dockerfile` and a matrix entry in
   `.github/workflows/build-ghcr-images.yml`.
4. **Re-derive the post-processing from the real HEF output** — list the
   vstream names/shapes on first inference, map heads by name (not by shape
   alone when two outputs share a shape), and confirm RGB/BGR and
   normalization. The Model Zoo per-model YAML documents the layout.
5. Update the module `README*.md` and `TEST_REPORT.md`.

The development SOP in `docs/CM5_HAILO8_MODEL_DEVELOPMENT_SOP_zh.md` codifies
the full checklist (fact-checking, HailoRT baseline, Docker, CI, AI Lab).

---

## Documentation

Per-module deep dives (build, CLI, troubleshooting, hardware verification):

- [CenterPose RegNetX-800MF](src/rpi5_hailo8_centerpose_regnetx_800mf/README.md) — [中文](src/rpi5_hailo8_centerpose_regnetx_800mf/README_zh.md)
- [STDC1](src/rpi5_hailo8_stdc1/README.md) — [中文](src/rpi5_hailo8_stdc1/README_zh.md)
- [CenterNet (resnet_v1_18)](src/rpi5_hailo8_centernet_resnet_v1_18_postprocess/README.md) — [中文](src/rpi5_hailo8_centernet_resnet_v1_18_postprocess/README_zh.md)
- [CenterNet (resnet_v1_50)](src/rpi5_hailo8_centernet_resnet_v1_50_postprocess/README.md) — [中文](src/rpi5_hailo8_centernet_resnet_v1_50_postprocess/README_zh.md)
- [DAMO-YOLO (tinynasL20_T)](src/rpi5_hailo8_damoyolo_tinynas_l20_t/README.md) — [中文](src/rpi5_hailo8_damoyolo_tinynas_l20_t/README_zh.md)
- [DAMO-YOLO (tinynasL25_S)](src/rpi5_hailo8_damoyolo_tinynas_l25_s/README.md) — [中文](src/rpi5_hailo8_damoyolo_tinynas_l25_s/README_zh.md)

Validation logs: each module ships a `TEST_REPORT.md` — hardware fields are
filled after the on-device run.

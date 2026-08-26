# reComputer-Hailo8-CV

[English] | [中文](./README_zh.md)

Industrial-grade computer-vision reference for **Raspberry Pi 5 / CM5 + Hailo-8**
(reComputer R Series). Each model ships as an independent module with a
FastAPI service: real-time MJPEG preview, REST prediction, USB camera, and
offline batch video analysis — built around the PCIe-attached Hailo-8
accelerator and HailoRT 4.23.x.

The repo covers three task families — **object detection** (CenterNet,
DAMO-YOLO, EfficientDet, NanoDet, SSD, Tiny-YOLO), **semantic segmentation**
(STDC1), and **pose estimation** (CenterPose). Every module follows the same
skeleton (HailoRT executor, letterbox + coordinate restore, frame buffer,
MJPEG encode); only the preprocessing, on-device post-processing mapping, and
decode differ per HEF. Some models use on-chip NMS (Hailo HPP, outputting
already-decoded boxes), while others (Tiny-YOLOv3/v4) output raw heads
requiring full CPU-side YOLOv3 decode.

---

## Hardware platform

| | |
|---|---|
| Board | Raspberry Pi 5 / CM5 (reComputer R Series carrier) |
| Accelerator | Hailo-8 M.2 (PCIe), device node `/dev/hailo0` |
| OS | Raspberry Pi OS Bookworm, kernel 6.12+ aarch64 |
| Host drivers | `hailort hailort-pcie-driver python3-hailort` (PCIe driver + firmware + Python API) |
| HailoRT | 4.23.x validated — host driver / firmware / container wheel **must share major.minor** |

---

## Included models (22)

| Model | Task | Parameters | Module | Container image |
|---|---|---:|---|---|
| CenterPose RegNetX-800MF | Pose (17 COCO keypoints) | 12.31M | `src/rpi5_hailo8_centerpose_regnetx_800mf/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/centerpose_regnetx_800mf:latest` |
| PaddleOCR v5 Mobile Detection | Text detection | 1.2M | `src/rpi5_hailo8_paddle_ocr_v5_mobile_detection/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/paddle_ocr_v5_mobile_detection:latest` |
| PaddleOCR v5 Mobile Recognition | Text recognition | — | `src/rpi5_hailo8_paddle_ocr_v5_mobile_recognition/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/paddle_ocr_v5_mobile_recognition:latest` |
| STDC1 | Semantic segmentation (Cityscapes 19) | 8.27M | `src/rpi5_hailo8_stdc1/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/stdc1:latest` |
| CenterNet (resnet_v1_18) | Object detection (COCO 80) | 14.22M | `src/rpi5_hailo8_centernet_resnet_v1_18_postprocess/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/centernet_resnet_v1_18_postprocess:latest` |
| CenterNet (resnet_v1_50) | Object detection (COCO 80) | 30.07M | `src/rpi5_hailo8_centernet_resnet_v1_50_postprocess/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/centernet_resnet_v1_50_postprocess:latest` |
| DAMO-YOLO (tinynasL20_T) | Object detection (COCO 80) | 11.35M | `src/rpi5_hailo8_damoyolo_tinynas_l20_t/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/damoyolo_tinynas_l20_t:latest` |
| DAMO-YOLO (tinynasL25_S) | Object detection (COCO 80) | 16.25M | `src/rpi5_hailo8_damoyolo_tinynas_l25_s/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/damoyolo_tinynas_l25_s:latest` |
| DAMO-YOLO (tinynasL35_M) | Object detection (COCO 80) | 33.98M | `src/rpi5_hailo8_damoyolo_tinynas_l35_m/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/damoyolo_tinynas_l35_m:latest` |
| EfficientDet-Lite0 | Object detection (COCO 80) | 3.56M | `src/rpi5_hailo8_efficientdet_lite0/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/efficientdet_lite0:latest` |
| EfficientDet-Lite1 | Object detection (COCO 80) | 4.73M | `src/rpi5_hailo8_efficientdet_lite1/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/efficientdet_lite1:latest` |
| EfficientDet-Lite2 | Object detection (COCO 80) | 5.93M | `src/rpi5_hailo8_efficientdet_lite2/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/efficientdet_lite2:latest` |
| NanoDet-RepVGG | Object detection (COCO 80) | 6.74M | `src/rpi5_hailo8_nanodet_repvgg/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/nanodet_repvgg:latest` |
| NanoDet-RepVGG-a12 | Object detection (COCO 80) | 5.13M | `src/rpi5_hailo8_nanodet_repvgg_a12/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/nanodet_repvgg_a12:latest` |
| NanoDet-RepVGG-a1-640 | Object detection (COCO 80) | 10.79M | `src/rpi5_hailo8_nanodet_repvgg_a1_640/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/nanodet_repvgg_a1_640:latest` |
| SSD MobileNet V1 | Object detection (COCO 80) | 6.79M | `src/rpi5_hailo8_ssd_mobilenet_v1/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/ssd_mobilenet_v1:latest` |
| SSD MobileNet V2 | Object detection (COCO 80) | 4.46M | `src/rpi5_hailo8_ssd_mobilenet_v2/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/ssd_mobilenet_v2:latest` |
| Tiny-YOLOv3 | Object detection (COCO 80) | 8.85M | `src/rpi5_hailo8_tiny_yolov3/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/tiny_yolov3:latest` |
| Tiny-YOLOv4 | Object detection (COCO 80) | 6.05M | `src/rpi5_hailo8_tiny_yolov4/` | `ghcr.io/seeed-projects/recomputer-hailo8-cv/tiny_yolov4:latest` |

All HEFs come from [Hailo Model Zoo v2.19.0](https://github.com/hailo-ai/hailo_model_zoo) (Hailo-8 target).

### Post-processing architecture

| Architecture | Models | On-chip NMS | Output format |
|---|---|---|---|
| On-chip NMS (HPP) | EfficientDet, NanoDet, SSD | Yes | Post-NMS tensor (Cx5xD) |
| On-chip max_finder | CenterNet | Partial | Sparse heatmap (128x128xC) |
| CPU YOLOv3 decode | Tiny-YOLOv3, Tiny-YOLOv4 | No | Raw heads (HxWx255) |
| CPU DFL decode | DAMO-YOLO | No | Raw nanodet_split heads |
| CPU 6-head decode | CenterPose | No | Raw CenterNet heads + keypoints |
| DB text detection | PaddleOCR v5 Mobile Detection | No | Text probability map |
| CTC text recognition | PaddleOCR v5 Mobile Recognition | No | CTC logits |

---

## Quick start (pre-built image)

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    ghcr.io/seeed-projects/recomputer-hailo8-cv/damoyolo_tinynas_l25_s:latest
```

Open `http://<device_IP>:8000` in a browser.

### 1. Host prep (one-time)

```bash
# Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh --mirror Aliyun
sudo systemctl enable docker && sudo systemctl start docker

# Hailo toolchain
sudo apt update
sudo apt install hailort hailort-pcie-driver python3-hailort
sudo reboot

# After reboot
hailortcli fw-control identify     # should report 4.23.x
ls /dev/hailo0
```

> Install `hailort hailort-pcie-driver python3-hailort` directly — NOT
> `hailo-all` (which can pull Hailo-10H / 5.x packages that don't match the
> Hailo-8 4.23.x baseline).

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

> **libhailort.so bind-mount**: the image ships only Python bindings; the
> native library comes from the host. Replace `4.23.0` with your firmware
> version if different.

---

## Repository layout

```text
reComputer-Hailo8-CV/
├── .github/workflows/build-ghcr-images.yml   # Per-model GHCR build (only changed models rebuild)
├── docker/hailo8/                             # One Dockerfile per model
│   ├── centerpose_regnetx_800mf.dockerfile
│   ├── paddle_ocr_v5_mobile_detection.dockerfile
│   ├── paddle_ocr_v5_mobile_recognition.dockerfile
│   ├── stdc1.dockerfile
│   ├── centernet_resnet_v1_18_postprocess.dockerfile
│   ├── centernet_resnet_v1_50_postprocess.dockerfile
│   ├── damoyolo_tinynas_l20_t.dockerfile
│   ├── damoyolo_tinynas_l25_s.dockerfile
│   ├── damoyolo_tinynas_l35_m.dockerfile
│   ├── efficientdet_lite0.dockerfile
│   ├── efficientdet_lite1.dockerfile
│   ├── efficientdet_lite2.dockerfile
│   ├── nanodet_repvgg.dockerfile
│   ├── nanodet_repvgg_a12.dockerfile
│   ├── nanodet_repvgg_a1_640.dockerfile
│   ├── ssd_mobilenet_v1.dockerfile
│   ├── ssd_mobilenet_v2.dockerfile
│   ├── tiny_yolov3.dockerfile
│   └── tiny_yolov4.dockerfile
└── src/
    ├── rpi5_hailo8_centerpose_regnetx_800mf/
    ├── rpi5_hailo8_paddle_ocr_v5_mobile_detection/
    ├── rpi5_hailo8_paddle_ocr_v5_mobile_recognition/
    ├── rpi5_hailo8_stdc1/
    ├── rpi5_hailo8_centernet_resnet_v1_18_postprocess/
    ├── rpi5_hailo8_centernet_resnet_v1_50_postprocess/
    ├── rpi5_hailo8_damoyolo_tinynas_l20_t/
    ├── rpi5_hailo8_damoyolo_tinynas_l25_s/
    ├── rpi5_hailo8_damoyolo_tinynas_l35_m/
    ├── rpi5_hailo8_efficientdet_lite0/
    ├── rpi5_hailo8_efficientdet_lite1/
    ├── rpi5_hailo8_efficientdet_lite2/
    ├── rpi5_hailo8_nanodet_repvgg/
    ├── rpi5_hailo8_nanodet_repvgg_a12/
    ├── rpi5_hailo8_nanodet_repvgg_a1_640/
    ├── rpi5_hailo8_ssd_mobilenet_v1/
    ├── rpi5_hailo8_ssd_mobilenet_v2/
    ├── rpi5_hailo8_tiny_yolov3/
    └── rpi5_hailo8_tiny_yolov4/

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

```bash
git clone https://github.com/Seeed-Projects/reComputer-Hailo8-CV.git
cd reComputer-Hailo8-CV/src/rpi5_hailo8_damoyolo_tinynas_l25_s

sudo docker build -f ../../docker/hailo8/damoyolo_tinynas_l25_s.dockerfile \
    -t hailo8-damoyolo-l25s:latest .

sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    hailo8-damoyolo-l25s:latest
```

---

## REST API

All endpoints on port `8000`; with `--net=host` reachable at
`http://<device_IP>:8000`. Replace `<slug>` with the model slug.

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/models/<slug>/predict` | POST | One-shot inference on uploaded image, specific video frame, or current camera frame |
| `/api/video_feed` | GET | MJPEG live stream with results overlaid (embed in an `<img>`) |
| `/api/config` | GET / POST | Read or update `obj_thresh` / `nms_thresh` |
| `/api/video/upload` | POST | Upload a video for batch analysis |
| `/api/video/analyze` | POST | Start an offline analysis job |
| `/api/video/status` | GET | Poll job progress |
| `/api/video/list` | GET | List uploaded sources and finished outputs |
| `/api/video/download/{filename}` | GET | Download an annotated output |

### Inference examples

```bash
# Image upload
curl -X POST http://<device_IP>:8000/api/models/damoyolo_tinynas_l25_s/predict -F "file=@test.jpg"

# Specific video frame (timestamp in seconds)
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

Embed the live stream:

```html
<img src="http://<device_IP>:8000/api/video_feed">
```

> Some models use on-chip NMS (HPP), others use CPU decode (YOLOv3, nanodet_split).
> The `nms_thresh` slider has effect on CPU-decode models; for on-chip-NMS models
> it's kept for API parity (NMS is already done on-device).

---

## Adapting to other models

1. Copy a module and rename (`rpi5_hailo8_<new_slug>/`).
2. Drop the new `.hef` into `model/` (lowercase slug name).
3. Add `docker/hailo8/<slug>.dockerfile` + a matrix entry in
   `.github/workflows/build-ghcr-images.yml`.
4. **Re-derive post-processing from the real HEF output** — check the Model Zoo
   YAML for the output layout, verify RGB/BGR and normalization on first
   inference (SOP §10).
5. Update `README*.md` and `TEST_REPORT.md`.

Full checklist: `docs/CM5_HAILO8_MODEL_DEVELOPMENT_SOP_zh.md`

---

## Documentation

- [CenterPose RegNetX-800MF](src/rpi5_hailo8_centerpose_regnetx_800mf/README.md) — [中文](src/rpi5_hailo8_centerpose_regnetx_800mf/README_zh.md)
- [STDC1](src/rpi5_hailo8_stdc1/README.md) — [中文](src/rpi5_hailo8_stdc1/README_zh.md)
- [CenterNet (resnet_v1_18)](src/rpi5_hailo8_centernet_resnet_v1_18_postprocess/README.md) — [中文](src/rpi5_hailo8_centernet_resnet_v1_18_postprocess/README_zh.md)
- [CenterNet (resnet_v1_50)](src/rpi5_hailo8_centernet_resnet_v1_50_postprocess/README.md) — [中文](src/rpi5_hailo8_centernet_resnet_v1_50_postprocess/README_zh.md)
- [DAMO-YOLO (tinynasL20_T)](src/rpi5_hailo8_damoyolo_tinynas_l20_t/README.md) — [中文](src/rpi5_hailo8_damoyolo_tinynas_l20_t/README_zh.md)
- [DAMO-YOLO (tinynasL25_S)](src/rpi5_hailo8_damoyolo_tinynas_l25_s/README.md) — [中文](src/rpi5_hailo8_damoyolo_tinynas_l25_s/README_zh.md)
- [DAMO-YOLO (tinynasL35_M)](src/rpi5_hailo8_damoyolo_tinynas_l35_m/README.md) — [中文](src/rpi5_hailo8_damoyolo_tinynas_l35_m/README_zh.md)
- [EfficientDet-Lite0](src/rpi5_hailo8_efficientdet_lite0/README.md) — [中文](src/rpi5_hailo8_efficientdet_lite0/README_zh.md)
- [EfficientDet-Lite1](src/rpi5_hailo8_efficientdet_lite1/README.md) — [中文](src/rpi5_hailo8_efficientdet_lite1/README_zh.md)
- [EfficientDet-Lite2](src/rpi5_hailo8_efficientdet_lite2/README.md) — [中文](src/rpi5_hailo8_efficientdet_lite2/README_zh.md)
- [NanoDet-RepVGG](src/rpi5_hailo8_nanodet_repvgg/README.md) — [中文](src/rpi5_hailo8_nanodet_repvgg/README_zh.md)
- [NanoDet-RepVGG-a12](src/rpi5_hailo8_nanodet_repvgg_a12/README.md) — [中文](src/rpi5_hailo8_nanodet_repvgg_a12/README_zh.md)
- [NanoDet-RepVGG-a1-640](src/rpi5_hailo8_nanodet_repvgg_a1_640/README.md) — [中文](src/rpi5_hailo8_nanodet_repvgg_a1_640/README_zh.md)
- [SSD MobileNet V1](src/rpi5_hailo8_ssd_mobilenet_v1/README.md) — [中文](src/rpi5_hailo8_ssd_mobilenet_v1/README_zh.md)
- [SSD MobileNet V2](src/rpi5_hailo8_ssd_mobilenet_v2/README.md) — [中文](src/rpi5_hailo8_ssd_mobilenet_v2/README_zh.md)
- [Tiny-YOLOv3](src/rpi5_hailo8_tiny_yolov3/README.md) — [中文](src/rpi5_hailo8_tiny_yolov3/README_zh.md)
- [Tiny-YOLOv4](src/rpi5_hailo8_tiny_yolov4/README.md) — [中文](src/rpi5_hailo8_tiny_yolov4/README_zh.md)

Validation logs: each module ships a `TEST_REPORT.md`.

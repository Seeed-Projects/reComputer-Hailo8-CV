# YOLO-World v2s on Raspberry Pi 5 + Hailo-8

Open-vocabulary object detection module for **Raspberry Pi 5 / CM5 + Hailo-8**
(reComputer R Series). It runs the Hailo Model Zoo YOLO-World v2s HEF with two
inputs: a 640x640 RGB frame and an 80x512 CLIP text-embedding matrix. The
bundled vocabulary is the 80 COCO classes.

> **Hailo-8 support status.** Hailo's public `HAILO8_*` model tables do not list
> zero-shot object detection, and the current Model Zoo network definition
> declares `supported_hw_arch: [hailo15h, hailo10h]`. The HEF is published under
> the official v2.20.0 `hailo8` prefix and is a real HEF v2 file, but this
> module has not been run on physical Hailo-8 hardware. Validate it on the board
> before treating the result as supported. No official Hailo-8
> `yolo_world_v2m.hef` is published, so this module ships the S variant only.

---

## 1. Features

- Real-time MJPEG preview with boxes, labels, and confidence scores.
- REST prediction for images, video frames, and the live camera frame.
- Offline batch video analysis with libx264 ultrafast encoding.
- JSON config endpoint for confidence and IoU display thresholds.
- 80-slot text vocabulary loaded from `model/coco_text_embeddings.npy` and
  `model/coco_classes.txt`.

The HEF contains the YOLO-World text-contrast head and a host-side NMS
post-process layer. The application supplies the CLIP embeddings once at
startup and reuses them for every frame.

---

## 2. Host preparation (Raspberry Pi 5)

The Hailo-8 PCIe driver, firmware, and userspace tools must be installed on the
host before the container can talk to the chip.

```bash
sudo apt update
sudo apt install hailo-all
sudo reboot

hailortcli fw-control identify
ls /dev/hailo0
```

Install Docker:

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh --mirror Aliyun
sudo systemctl enable docker
sudo systemctl start docker
```

The container's `hailort` wheel major.minor must match the host firmware and
`libhailort.so` major.minor. The bundled wheel is
`hailort-4.23.0-cp311-cp311-linux_aarch64.whl`.

---

## 3. Build and run

```bash
cd src/rpi5_hailo8_yolo_world_v2s
sudo docker build -f ../../docker/hailo8/yolo_world_v2s.dockerfile \
  -t rpi5-hailo8-yolo-world-v2s:latest .

sudo docker run --rm --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  rpi5-hailo8-yolo-world-v2s:latest
```

The default command loops `video/test.mp4` and serves the Web UI on
`http://<Board_IP>:8000`.

Camera mode (override the default command):

```bash
sudo docker run --rm --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  --device /dev/video0:/dev/video0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  rpi5-hailo8-yolo-world-v2s:latest \
  python web_detection.py --model_path model/yolo_world_v2s.hef --camera_id 0
```

Replace `4.23.0` with the version reported by `hailortcli fw-control identify`.

---

## 4. Model assets

| File | Size | Source |
|---|---:|---|
| `model/yolo_world_v2s.hef` | 22,431,677 B | [Model Zoo v2.20.0 hailo8](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/yolo_world_v2s.hef) |
| `model/coco_text_embeddings.npy` | 163,968 B | CLIP ViT-B/32 text embeddings, shape `(1, 80, 512)` |
| `model/coco_classes.txt` | 701 B | 80 COCO class names, one per line |

The HEF SHA256 is
`39b5b17df32c11503aa9a9b5d7d51ad0ee77265b0fb799a73aa01857bb33c6ec`.
See [model/README.md](model/README.md) for the full provenance and caveats.

---

## 5. Vocabulary

YOLO-World's HEF accepts exactly 80 text prompts. Row `i` of
`coco_text_embeddings.npy` produces output class `i`, and row `i` of
`coco_classes.txt` is the label drawn for that class.

The bundled embeddings were generated with
`openai/clip-vit-base-patch32` (`CLIPTextModelWithProjection`, class name only,
L2-normalized), matching Hailo's
`create_coco_tfrecord_with_text_embeddings.py`.

To use another vocabulary, precompute exactly 80 embeddings of the same shape
and pass both files:

```bash
python web_detection.py \
  --model_path model/yolo_world_v2s.hef \
  --embeddings_path /path/to/my_embeddings.npy \
  --class_path /path/to/my_classes.txt
```

The unused slots still need valid prompts; an all-zero row is not a safe
"empty" class because the contrast head can emit a spurious mid-confidence
score. Pad your vocabulary to 80 rows with meaningful prompts.

---

## 6. HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Web UI |
| GET/POST | `/api/config` | Read/update display thresholds |
| POST | `/api/video/upload` | Upload a video |
| GET | `/api/video/list` | List uploads and outputs |
| POST | `/api/video/analyze` | Start offline analysis |
| GET | `/api/video/status` | Progress and error state |
| GET | `/api/video/download/{filename}` | Download an annotated MP4 |
| POST | `/api/models/yolo_world/predict` | Single-frame prediction |

Example:

```bash
curl -X POST "http://<Board_IP>:8000/api/models/yolo_world/predict" \
  -F "file=@frame.jpg"
```

The response contains the label, confidence, and pixel coordinates for each
detection.

---

## 7. Implementation notes

- Inputs: `yolo_world_v2s/input_layer1` (640x640x3 RGB, UINT8) and
  `yolo_world_v2s/input_layer2` (80x512 float32 CLIP embeddings). Input names
  are discovered from the HEF at startup, not hard-coded by position.
- The image is letterboxed to 640x640, converted to RGB, and fed without manual
  normalization; the HEF's normalization layer divides by 255.
- Boxes are un-letterboxed back to the source frame before drawing.
- The HEF's NMS layer uses the Model Zoo config: 80 classes, IoU 0.7, up to 300
  proposals per class, score threshold 0.001. The UI's confidence slider filters
  the returned rows; the NMS itself is not re-run on the host.

---

## 8. Known limitations

- The vocabulary is fixed at startup. This module does not run a CLIP text
  encoder on the board, so prompts cannot be changed from the UI.
- The H8 HEF is not listed in Hailo's public H8 model tables. Treat the module
  as unverified until it passes on-board testing.
- No H8 v2m HEF is published; the M variant is available on Hailo-10H only.
- The Web service is intended for a trusted LAN and has no authentication or
  TLS by default.
- Real-time line-by-line accuracy is not characterized; see `TEST_REPORT.md` for
  the checks that remain pending.
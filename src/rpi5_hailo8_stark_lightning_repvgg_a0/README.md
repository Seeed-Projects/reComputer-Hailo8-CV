# STARK-Lightning on Raspberry Pi 5 + Hailo-8

Single-object tracking module for **Raspberry Pi 5 / CM5 + Hailo-8**
(reComputer R Series). It runs the official Hailo Model Zoo
`stark_lightning_repvgg_a0` HEF with two inputs: a 128x128 template crop and a
320x320 search crop.

Unlike the detection modules, tracking is stateful. The first frame seeds the
template from a user-selected box; each later frame crops the search region
around the previous prediction, runs the HEF, decodes the corner heatmaps, and
maps the resulting box back to image coordinates.

---

## 1. Host preparation (Raspberry Pi 5)

```bash
sudo apt update
sudo apt install hailo-all
sudo reboot

hailortcli fw-control identify
ls /dev/hailo0

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh --mirror Aliyun
sudo systemctl enable docker
sudo systemctl start docker
```

The bundled wheel is `hailort-4.23.0-cp311-cp311-linux_aarch64.whl`; the host
firmware and `libhailort.so` must share the same major.minor.

---

## 2. Build and run

```bash
cd src/rpi5_hailo8_stark_lightning_repvgg_a0
sudo docker build -f ../../docker/hailo8/stark_lightning_repvgg_a0.dockerfile \
  -t rpi5-hailo8-stark:latest .

sudo docker run --rm --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  rpi5-hailo8-stark:latest
```

Open `http://<Board_IP>:8000`.

1. Upload a video.
2. The first frame is shown on a canvas. Drag a box around the target.
3. Click **Run tracking**. The annotated MP4 appears when processing finishes.

Replace `4.23.0` with the version reported by `hailortcli fw-control identify`.

---

## 3. Model assets

| File | Size | Source |
|---|---:|---|
| `model/stark_lightning_repvgg_a0.hef` | 4,425,622 B | Model Zoo v2.20.0 hailo8 |
| `hailort-packages/hailort-4.23.0-...whl` | 9,939,376 B | HailoRT 4.23.0 |

HEF SHA256:
`7319ef5e0016e85510a71a86e8231a1fb0558a6475fa041e08919cc960c03299`.

Official Hailo-8 metrics: 56.9 float / 55.2 hardware Success-plot AUC on LaSOT,
1074 FPS, 4.62M parameters, 3.48G operations. See
[model/README.md](model/README.md) for provenance.

---

## 4. How tracking works

The implementation follows the Model Zoo `tracker_infer.py` closed loop:

1. `initialize(frame, box)`: crop a square region of `2x` the target area,
   resize to 128x128, and store it as the template.
2. `update(frame)`: crop a square region of `5x` the previous box area around
   the previous center, resize to 320x320, run the HEF with
   `template + search`.
3. Decode the HEF's two 20x20 corner heatmaps with a per-channel softmax and
   soft-argmax on the 320x320 grid, producing normalized `xyxy`; multiply by
   the search-crop scale, add the crop offset, and clip to the frame.

The HEF applies normalization in-network, so the application feeds raw RGB
pixels as float32; no manual mean/std subtraction.

---

## 5. HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Web UI |
| POST | `/api/video/upload` | Upload a video |
| GET | `/api/video/list` | List uploads and outputs |
| GET | `/api/video/first_frame?filename=...` | JPEG of the first frame |
| POST | `/api/video/analyze` | Start tracking (`filename`, `x`, `y`, `w`, `h`) |
| POST | `/api/video/analyze/stop` | Request cancellation |
| GET | `/api/video/status` | Progress, output name, error |
| GET | `/api/video/download/{filename}` | Download the annotated MP4 |

Example:

```bash
curl -X POST "http://<Board_IP>:8000/api/video/analyze" \
  -F "filename=demo.mp4" -F "x=420" -F "y=180" -F "w=160" -F "h=240"
```

---

## 6. Known limitations

- The target box must be selected in the first frame. There is no
  re-initialization or multi-object tracking.
- Tracking is offline over uploaded video; this module does not expose a live
  camera MJPEG preview.
- The official Hailo-8 HEF is used, but this module has not been run on a
  physical board here. Hardware validation is pending; see `TEST_REPORT.md`.
- The Web service targets a trusted LAN and has no authentication or TLS.
- Output videos have no audio; odd dimensions are cropped by at most one row or
  column for encoding.
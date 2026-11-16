# Test Report - ByteTrack-M (MOT17) on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | bytetrack_m_mot17 |
| Task | MOT17 pedestrian detection + ByteTrack association |
| Parameters | 25.28M |
| Operations | 206.12G |
| MOTA (float / Hailo-8) | 86.2 / 86.1 |
| Model Zoo version | v2.20.0 |
| Compile target | hailo8 |

## HEF artifact

| Field | Value |
|---|---|
| File | model/bytetrack_m_mot17.hef |
| Size | 81,354,896 bytes |
| SHA256 | `68a4586814608bcdda613cc625bedc379008b0d3e95bc32712e8411f60f92dd8` |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/bytetrack_m_mot17.hef |

## Input / output

| Field | Value |
|---|---|
| Input | 608x1088x3 RGB uint8 (letterbox, padding 114) |
| On-chip NMS | NO (9 raw heads) |
| Output | 3 strides x (box 4, obj 1, class 1): 19x34, 38x68, 76x136 |
| Classes | 1 (person) |
| Decode | YOLOX-style center/size decode, score=obj*class, NMS 0.7 |
| Tracking | ByteTrack (Kalman + IoU matching), host-side |

## Verification status

| Check | Status |
|---|---|
| Python syntax (py_compile) | Pass |
| Offline nine-head decode + tracker ID persistence | Pass |
| Configuration defaults (score 0.1, NMS 0.7, track 0.5) | Pass |
| Dockerfile present + CMD references real HEF | Pass |
| CI matrix entry added | Pass |

### Hardware verification - pending

| Check | Status |
|---|---|
| HEF loads, output head shapes confirmed | Pending |
| Demo video: person boxes and stable track IDs | Pending |
| Single-image REST API | Pending |
| Official GHCR image re-pull + run | Pending |

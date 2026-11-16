# Test Report - ByteTrack-S (MOT17) on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | bytetrack_s_mot17 |
| Task | MOT17 pedestrian detection + ByteTrack association |
| Parameters | 8.94M |
| Operations | 43.21G |
| MOTA (float / Hailo-8) | 79.5 / 77.7 |
| Model Zoo version | v2.20.0 |
| Compile target | hailo8 |

## HEF artifact

| Field | Value |
|---|---|
| File | model/bytetrack_s_mot17.hef |
| Size | 14,836,099 bytes |
| SHA256 | `d0165ac42ed585a71c7acbe82310175e813be1b77bf6f3d39ce0b327af804e3a` |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.20.0/hailo8/bytetrack_s_mot17.hef |

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

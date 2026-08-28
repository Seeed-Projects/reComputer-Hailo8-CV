# Test report — tiny_yolov4_license_plates

- Date: 2026-08-28
- Hardware target: CM5 / Raspberry Pi 5 + Hailo-8
- Runtime target: HailoRT 4.23.x, Python 3.11 aarch64
- HEF: `tiny_yolov4_license_plates.hef`, 8,208,457 bytes
- HEF SHA256: `97EF984739960D6AA6A44140E486565C5EFB46BD3DC28790C5A786FEF6683B12`
- Video: official Hailo `lpr_ayalon.mp4`, transcoded to 1280x720 H.264, 33.17 s
- Video SHA256: `72063C7C0190E172DA0740461B28DA1EE17AA89860C3682DCB0B908559ED86E8`

## Completed locally

- Python AST syntax check
- HEF/video download integrity and media probing
- Static verification of input, output, anchors, labels, and API path against Model Zoo v2.17/TAPPAS v3.29

## Pending hardware validation

- HEF load and real vstream names/shapes
- Video, image API, offline analysis, and USB camera inference
- Box alignment and confidence calibration
- Docker build/run on arm64

No hardware-pass claim is made.

# Test report — lprnet pipeline

- Date: 2026-08-28
- Hardware target: CM5 / Raspberry Pi 5 + Hailo-8
- Runtime target: HailoRT 4.23.x, Python 3.11 aarch64
- LPRNet HEF SHA256: `F6AD6E29F245B1EC174F67D2600B382FE5336548554CC040C92FF96959860BEA`
- Detector HEF SHA256: `97EF984739960D6AA6A44140E486565C5EFB46BD3DC28790C5A786FEF6683B12`
- Video: official Hailo `lpr_ayalon.mp4`, transcoded to 1280x720 H.264, 33.17 s

## Completed locally

- Python AST syntax check
- Static verification of Hailo's OCR algorithm: mean, softmax, CTC collapse, confidence and length gates
- Shared VDevice round-robin scheduler configured for both HEFs
- Model/video hashes and media probing

## Pending hardware validation

- Compatibility of both HEFs with HailoRT/firmware 4.23.x on the target board
- Real detector and OCR vstream names/shapes
- End-to-end recognition accuracy and scheduler stability
- Image API, video, offline analysis, camera, and Docker tests

No hardware-pass claim is made.

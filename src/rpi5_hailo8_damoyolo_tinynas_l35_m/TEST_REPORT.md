# Test Report — DAMO-YOLO (tinynasL35_M) on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | damoyolo_tinynasL35_M (slug: damoyolo_tinynas_l35_m) |
| Task | Object detection (COCO 80 classes) |
| Backbone | TinyNAS-L35, Medium size |
| Parameters | 33.98M |
| Operations | 61.64G |
| Framework source | pytorch (DAMO-YOLO) |
| Model Zoo version | v2.19.0 |
| Compile target | hailo8 |
| Family | damoyolo (variants: L20_T, L25_S, L35_M) |

## HEF artifact

| Field | Value |
|---|---|
| File | model/damoyolo_tinynas_l35_m.hef |
| Size | ~27 MB (27,513,105 bytes) |
| SHA256 | 741ed41085fc17ec47a5aa8dcaffd076e6d043a8b75a8b2785ef060b577147eb |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/damoyolo_tinynasL35_M.hef |
| Note | Renamed from Model Zoo name (damoyolo_tinynasL35_M) to lowercase slug for GHCR validity |

## Input / output (from Model Zoo network YAML + base/damoyolo.yaml)

| Field | Value |
|---|---|
| Input | 640x640x3 RGB, uint8 (normalize_in_net mean=0/std=1 = no-op) |
| Padding | color 0 (black) |
| Output | 6 tensors: 80x80x68, 80x80x81, 40x40x68, 40x40x81, 20x20x68, 20x20x81 |
| Per scale | box 68ch (4 x 17 DFL) + cls 81ch (80 classes + 1, take 80) |
| Output dtype | FLOAT32 |
| Strides | 8, 16, 32 |
| regression_length | 16 (-> 4 x 17 = 68 box channels) |
| On-chip | sigmoid (classification only) |
| meta_arch | nanodet_split |

## Runtime baseline

| Field | Value |
|---|---|
| HailoRT | 4.23.x |
| Python | 3.11, aarch64 |
| Wheel | hailort-4.23.0-cp311-cp311-linux_aarch64.whl |
| Kernel module | hailo_pci |
| Device node | /dev/hailo0 |

## Verification status

This report covers the static/code-level verification performed on the
development host (no Hailo-8 hardware available locally). Hardware fields are
left for the on-device run.

| Check | Status |
|---|---|
| Python syntax (py_compile) | Pass |
| Module structure matches SOP | Pass |
| Dockerfile present + CMD references real HEF | Pass |
| CI matrix entry added | Pass |
| HEF input/output taken from Model Zoo source (not guessed) | Pass |
| First-inference vstream logging implemented (SOP §10) | Pass |
| Post-processing identical to verified L20_T / L25_S builds (same base config) | Pass |

### Hardware verification — pending

To be filled after the SSH pull-and-run on CM5 + Hailo-8:

| Check | Status |
|---|---|
| PCI device / `/dev/hailo0` present | Pending |
| `hailortcli fw-control identify` succeeds | Pending |
| HEF loads, no driver/firmware mismatch | Pending |
| Output vstream names/shapes match YAML (6 tensors) | Pending |
| Per-scale box/cls grouping confirmed from first-inference log | Pending |
| Demo video: boxes align with objects, correct classes | Pending |
| RGB vs BGR channel order correct (boxes not misaligned/misclassified) | Pending |
| Single-image REST API: class/confidence/box correct | Pending |
| Offline video analysis: upload/progress/download | Pending |
| USB camera: real-time preview, latency, stability | Pending |
| Official GHCR image re-pull + run | Pending |

## Known points to verify on hardware

- Same as L20_T / L25_S: six heads grouped by spatial size + channel count
  (fewer ch = box 68ch, more = cls 81ch), robust to HailoRT dict ordering.
  First-inference log prints every vstream name/shape and the per-scale grouping;
  verify it matches [box80,cls80,box40,cls40,box20,cls20].
- The cls head is expected to be 81 channels (80 classes + 1 dropped); if a
  build returns 80, the decoder already handles both.
- Channel order: RGB assumed (DAMO-YOLO/Model Zoo convention); if the street
  scene shows systematically wrong classes or misaligned boxes, swap BGR/RGB.
- The decoder treats cls as already-sigmoid'd (on-chip). If boxes/scores look
  like raw logits, the on-chip sigmoid may not have fired — the decoder would
  need a CPU sigmoid fallback.

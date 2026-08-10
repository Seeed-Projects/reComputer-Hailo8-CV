# Test Report — CenterNet (resnet_v1_50_postprocess) on CM5 + Hailo-8

## Model

| Field | Value |
|---|---|
| Model name | centernet_resnet_v1_50_postprocess |
| Task | Object detection (COCO 80 classes) |
| Backbone | ResNet-50 |
| Parameters | 30.07M |
| Operations | 56.92G |
| Framework source | gluoncv |
| Model Zoo version | v2.19.0 |
| Compile target | hailo8 |
| Sibling | centernet_resnet_v1_18_postprocess (same I/O contract, lighter backbone) |

## HEF artifact

| Field | Value |
|---|---|
| File | model/centernet_resnet_v1_50_postprocess.hef |
| Size | ~30.6 MB (30,564,239 bytes) |
| SHA256 | b2c31d179f564f763af3b6a2c78d1848c9c463594ee7c8f67ef208c32d10bb54 |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/centernet_resnet_v1_50_postprocess.hef |

## Input / output (from Model Zoo network YAML)

| Field | Value |
|---|---|
| Input | 512x512x3 RGB, uint8 (normalization compiled into HEF) |
| Normalization | mean [123.675, 116.28, 103.53], std [58.395, 57.12, 57.375], RGB |
| Output 0 | `centernet0_conv3` — wh, 128x128x2 (box w/h, stride units) |
| Output 1 | `centernet0_conv5` — reg, 128x128x2 (sub-pixel center offset) |
| Output 2 | `threshold_confidence/.../Relu` — sparse heatmap, 128x128x80 (on-chip max_finder + threshold 0.2) |
| Output dtype | FLOAT32 |
| Device pre-post layers | max_finder=true |

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
| Post-processing identical to verified v1_18 build | Pass |

### Hardware verification — pending

To be filled after the SSH pull-and-run on CM5 + Hailo-8:

| Check | Status |
|---|---|
| PCI device / `/dev/hailo0` present | Pending |
| `hailortcli fw-control identify` succeeds | Pending |
| HEF loads, no driver/firmware mismatch | Pending |
| Output vstream names/shapes match YAML | Pending |
| hm/wh/reg mapping confirmed from first-inference log | Pending |
| Demo video: boxes align with objects, no offset | Pending |
| Single-image REST API: class/confidence/box correct | Pending |
| Offline video analysis: upload/progress/download | Pending |
| USB camera: real-time preview, latency, stability | Pending |
| Official GHCR image re-pull + run | Pending |

## Known points to verify on hardware

- The two 2-channel heads (wh, reg) are mapped by conv number when the vstream
  name exposes it (conv3 -> wh, conv5 -> reg), otherwise by network output
  order. The first-inference log prints both, so swap wh/reg if the on-device
  names indicate the opposite order.
- The heatmap is assumed sparse (on-chip max_finder). The defensive 3x3
  local-max suppression is idempotent if sparse; if the log shows a dense
  heatmap, the suppression still handles it.

# LPRNet Pipeline on CM5 + Hailo-8

This module runs a two-network pipeline: Tiny-YOLOv4 detects a license plate in the full frame, then LPRNet recognizes digits from the original-resolution crop. Both models share the Hailo-8 through the HailoRT round-robin scheduler and shared VDevice group.

## Model contract

| Stage | Input | Output | Post-process |
|---|---|---|---|
| `tiny_yolov4_license_plates` | 416x416x3 RGB | two raw YOLO heads | sigmoid + anchors + CPU NMS |
| `lprnet` | 300x75x3 BGR crop | 5x19x11 logits | mean over axis 0 + softmax + CTC greedy decode |

The OCR alphabet is `0123456789-`. A result is accepted when mean character confidence is at least 0.90 and at least seven digits remain after CTC collapse. This official model does **not** recognize letters or Chinese province characters.

Before OCR, the module applies the official TAPPAS central-crop Laplacian variance gate (`quality >= 100`).

## Demo video

The bundled video is a 1280x720 H.264 transcode of Hailo's official TAPPAS v3.29 `lpr_ayalon.mp4`:

https://hailo-tappas.s3.eu-west-2.amazonaws.com/v3.29/general/media/lpr_ayalon.mp4

## Build and run

```bash
sudo docker build -f docker/hailo8/lprnet.dockerfile \
  -t lprnet:latest src/rpi5_hailo8_lprnet

sudo docker run --rm --name cm5-hailo8-lprnet --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  lprnet:latest
```

Open `http://<Board_IP>:8000`.

## REST API

```bash
curl -X POST "http://<Board_IP>:8000/api/models/lprnet/predict" \
  -F "file=@traffic.jpg"
```

Each prediction includes the detector confidence and box, quality score, OCR candidate/confidence, and the accepted `plate` value (or `null`).

## Validation status

Python syntax and source/asset checks are complete. Two-HEF scheduling, real tensor shapes, OCR output, Docker build, and end-to-end results still require CM5 + Hailo-8 validation.

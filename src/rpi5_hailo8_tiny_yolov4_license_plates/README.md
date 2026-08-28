# Tiny-YOLOv4 License Plate Detection on CM5 + Hailo-8

Single-class license-plate detection using the official Hailo Model Zoo v2.17 Hailo-8 HEF. The app supports the bundled video, USB camera, image/video REST prediction, MJPEG preview, and offline video analysis.

## Model contract

| Item | Value |
|---|---|
| Input | 416x416x3 RGB uint8; normalization is compiled into the HEF |
| Output | 13x13x18 and 26x26x18 raw Tiny-YOLOv4 heads |
| Class | `license_plate` |
| Post-process | sigmoid, anchor/grid decode, score filtering, CPU NMS |
| Model Zoo metric | 74.083 mAP (internal license-plate dataset) |

The anchors are taken from the v2.17 network YAML. Both combined 18-channel outputs and legacy split Hailo-8 output tensors are accepted.

## Demo video

The bundled `video/test.mp4` is a 1280x720 H.264 transcode of Hailo TAPPAS v3.29 `lpr_ayalon.mp4`, the official default LPR demo source:

https://hailo-tappas.s3.eu-west-2.amazonaws.com/v3.29/general/media/lpr_ayalon.mp4

## Build and run

```bash
sudo docker build -f docker/hailo8/tiny_yolov4_license_plates.dockerfile \
  -t tiny_yolov4_license_plates:latest \
  src/rpi5_hailo8_tiny_yolov4_license_plates

sudo docker run --rm --name cm5-hailo8-lp-detector --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.4.23.0:/usr/lib/libhailort.so.4.23.0:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  tiny_yolov4_license_plates:latest
```

Open `http://<Board_IP>:8000`.

## REST API

```bash
curl -X POST "http://<Board_IP>:8000/api/models/tiny_yolov4_license_plates/predict" \
  -F "file=@test.jpg"
```

The response contains `class`, `confidence`, and an `xyxy` pixel box. Camera mode uses `--camera_id 0`; video analysis endpoints follow the repository-wide API contract.

## Validation status

Python syntax and static model metadata are checked locally. CM5 + Hailo-8 inference, output tensor logging, Docker build, camera operation, and final box alignment still require hardware validation.

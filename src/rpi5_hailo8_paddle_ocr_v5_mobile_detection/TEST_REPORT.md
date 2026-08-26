# Test report

## Local static validation

- Python syntax: `web_detection.py`, `py_utils/hailo_executor.py`, and `py_utils/db_postprocess.py` compile successfully.
- Bundled Hailo-8 detection HEF is present (5,486,955 bytes).
- `video/test.mp4` is a 6-second H.264 demo generated from the self-authored English document image.

## Hardware validation required

Run the Docker command from `README.md` on CM5 + Hailo-8 with HailoRT 4.23.x.
Confirm `/dev/hailo0`, open the MJPEG page, then POST `video/test.png` to the REST endpoint.

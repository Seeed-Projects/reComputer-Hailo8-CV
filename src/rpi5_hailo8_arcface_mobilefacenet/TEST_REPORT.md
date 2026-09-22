# ArcFace MobileFaceNet validation checklist

Static checks completed during development:

- Python source compiles successfully (`python -m py_compile web_detection.py py_utils/*.py`).
- The container uses Python 3.11 and HailoRT 4.23.0 for aarch64.
- Default HEF path is `model/arcface_mobilefacenet.hef`.
- HEF identity: 4,100,573 bytes,
  SHA-256 `3c13c23b72fef261c998a31a3628bbe8b223ffbbfd4ed405b797f7c27d948adf`,
  compiled for Hailo-8 from Hailo Model Zoo v2.19.0.
- Preprocessing letterboxes to 112×112 and converts BGR to RGB; normalization
  is inside the HEF.
- Post-processing selects the `fc1` embedding vstream and returns a 512-element
  float32 vector, rejecting empty or non-finite tensors.
- REST route is `/api/models/arcface_mobilefacenet/predict` and returns
  `embedding` + `dimension`.
- No CenterPose detection/pose code or class label tables remain in the module.

Hardware acceptance on Raspberry Pi 5 + Hailo-8:

1. Confirm `hailortcli --version` reports 4.23.x and `/dev/hailo0` exists.
2. Build or pull the arm64 container.
3. Check that the first inference logs the input vstream
   `arcface_mobilefacenet/input_layer1` and the embedding output (expected
   `arcface_mobilefacenet/fc1`, 512 values).
4. Confirm the demo video plays with the "ArcFace embedding: 512D" overlay.
5. Send a face image to the prediction endpoint and check that the response
   contains 512 finite floats.
6. Verify similarity: two images of the same face should score higher than two
   different faces (client-side cosine similarity).
7. Test camera mode with `/dev/video0` mounted.

Status: static checks pass; hardware steps above are pending an on-device run.
Static compilation is not a substitute for the CM5 + Hailo-8 acceptance test.
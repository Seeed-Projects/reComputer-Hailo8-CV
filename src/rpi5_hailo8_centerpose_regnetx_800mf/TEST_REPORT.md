# CenterPose RegNetX-800MF validation checklist

Static checks completed during development:

- Python source compiles successfully.
- The container uses Python 3.11 and HailoRT 4.23.0 for aarch64.
- Default HEF path is `model/centerpose_regnetx_800mf.hef`.
- Input size and CenterPose output-head mapping are covered by decoder tests.
- REST route is `/api/models/centerpose_regnetx_800mf/predict`.

Hardware acceptance on Raspberry Pi 5 + Hailo-8:

1. Confirm `hailortcli --version` reports 4.23.x and `/dev/hailo0` exists.
2. Build or pull the arm64 container.
3. Check that the first inference logs six 128×128 outputs with channel counts
   1, 2, 34, 2, 17, and 2.
4. Confirm the demo video shows person boxes and COCO skeletons.
5. Send an image to the prediction endpoint and validate the JSON response.
6. Test camera mode with `/dev/video0` mounted.


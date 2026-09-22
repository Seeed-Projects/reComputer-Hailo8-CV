# ArcFace MobileFaceNet HEF

Place `arcface_mobilefacenet.hef` in this directory.

- Source: Hailo Model Zoo compiled models v2.19.0
- URL: https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/arcface_mobilefacenet.hef
- Hardware architecture: Hailo-8 (not Hailo-8L or Hailo-10H)
- Input: 112×112×3
- Output: 512-dim face embedding (`arcface_mobilefacenet/fc1`)
- Expected size: 4,100,573 bytes
- SHA-256: `3c13c23b72fef261c998a31a3628bbe8b223ffbbfd4ed405b797f7c27d948adf`

The HEF contains an in-net normalization layer, so the application feeds raw
uint8 pixels and does not normalize on the host.
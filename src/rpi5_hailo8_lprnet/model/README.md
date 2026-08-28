# Model files

This pipeline uses two official Hailo-8 HEFs:

| File | Source version | Size | SHA256 |
|---|---:|---:|---|
| `lprnet.hef` | Model Zoo v2.16 | 7,469,917 bytes | `F6AD6E29F245B1EC174F67D2600B382FE5336548554CC040C92FF96959860BEA` |
| `tiny_yolov4_license_plates.hef` | Model Zoo v2.17 | 8,208,457 bytes | `97EF984739960D6AA6A44140E486565C5EFB46BD3DC28790C5A786FEF6683B12` |

Sources:

- https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/lprnet.hef
- https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8/tiny_yolov4_license_plates.hef

LPRNet accepts a 300x75 BGR plate crop. Its HEF performs BGR-to-RGB conversion and normalization on-device. The 5x19x11 output represents digits 0-9 plus the CTC blank class; it does not contain letters or Chinese province characters.

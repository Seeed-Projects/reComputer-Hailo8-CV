# HailoRT Python wheel directory

Place a `hailort-<version>-cp311-cp311-linux_aarch64.whl` here. The Dockerfile
installs whatever it finds matching `hailort-*.whl`. The major.minor version
must match the board firmware and `libhailort.so`.

import threading

import numpy as np

from hailo_platform import (
    HEF,
    VDevice,
    FormatType,
    HailoStreamInterface,
    ConfigureParams,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
)


class HailoInfer:
    """Persistent HailoRT 4.23 inference session for a Hailo-8 HEF."""

    def __init__(self, hef_path):
        self.hef = HEF(hef_path)
        self.target = VDevice()
        self._lock = threading.Lock()
        self._released = False

        cfg = ConfigureParams.create_from_hef(
            hef=self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.target.configure(self.hef, cfg)[0]
        self.ng_params = self.network_group.create_params()

        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.output_infos = self.hef.get_output_vstream_infos()
        shape = self.input_info.shape
        if len(shape) == 4:
            self.input_h, self.input_w = shape[1], shape[2]
        else:
            self.input_h, self.input_w = shape[0], shape[1]

        in_params = InputVStreamParams.make(
            self.network_group, format_type=FormatType.UINT8
        )
        out_params = OutputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32
        )

        # Keep activation and pipeline alive for the process lifetime.
        self._activation = self.network_group.activate(self.ng_params)
        self._activation.__enter__()
        self._pipeline_ctx = InferVStreams(
            self.network_group, in_params, out_params
        )
        self._pipeline = self._pipeline_ctx.__enter__()

        print(f"Face Landmarks Lite HEF loaded: {hef_path}")
        print(
            f"  Input:  {self.input_info.name} "
            f"shape={self.input_info.shape}"
        )
        for info in self.output_infos:
            print(f"  Output: {info.name} shape={info.shape}")

    def run(self, image):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        with self._lock:
            if self._released:
                raise RuntimeError("HailoInfer has already been released")
            try:
                return self._pipeline.infer({self.input_info.name: image})
            except Exception as exc:
                print(f"Hailo inference error: {exc}")
                return None

    def release(self):
        with self._lock:
            if self._released:
                return
            try:
                self._pipeline_ctx.__exit__(None, None, None)
            except Exception:
                pass
            try:
                self._activation.__exit__(None, None, None)
            except Exception:
                pass
            try:
                self.target.release()
            except Exception:
                pass
            self._released = True
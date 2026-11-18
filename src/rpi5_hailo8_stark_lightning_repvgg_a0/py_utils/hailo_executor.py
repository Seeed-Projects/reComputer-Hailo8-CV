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
    """HailoRT 4.23 session for STARK-Lightning.

    The HEF has two image inputs:
      * input_layer1: 128x128x3 template (exemplar) crop
      * input_layer2: 320x320x3 search crop

    Both are fed as float32 RGB in 0-255. HailoRT applies the vstream
    quantization and the HEF's in-net normalization (mean [123.675, 116.28,
    103.53], std [58.395, 57.12, 57.375]).
    """

    def __init__(self, hef_path):
        self.hef = HEF(hef_path)
        self.target = VDevice()

        cfg = ConfigureParams.create_from_hef(
            hef=self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.target.configure(self.hef, cfg)[0]
        self.ng_params = self.network_group.create_params()

        self.input_infos = list(self.hef.get_input_vstream_infos())
        self.template_input = None
        self.search_input = None
        for info in self.input_infos:
            name = info.name.lower()
            if "input_layer1" in name:
                self.template_input = info
            elif "input_layer2" in name:
                self.search_input = info

        # Fallback for renamed streams: input_layer1 is the 128 template,
        # input_layer2 is the 320 search.
        if self.template_input is None or self.search_input is None:
            for info in self.input_infos:
                shape = tuple(int(v) for v in info.shape)
                if 320 in shape and self.search_input is None:
                    self.search_input = info
                elif 128 in shape and self.template_input is None:
                    self.template_input = info

        if self.template_input is None or self.search_input is None:
            raise ValueError(
                "STARK HEF must expose a 128x128 template input and a 320x320 "
                "search input: "
                + ", ".join(f"{i.name}{tuple(i.shape)}" for i in self.input_infos)
            )

        self._template_shape = tuple(int(v) for v in self.template_input.shape)
        self._search_shape = tuple(int(v) for v in self.search_input.shape)
        self.output_infos = self.hef.get_output_vstream_infos()

        # Float32 host buffers: the HEF owns quantization and normalization.
        in_p = InputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32
        )
        out_p = OutputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32
        )

        # Hold activation + pipeline for the lifetime of the wrapper.
        self._act = self.network_group.activate(self.ng_params)
        self._act.__enter__()
        self._pipe_ctx = InferVStreams(self.network_group, in_p, out_p)
        self._pipe = self._pipe_ctx.__enter__()

        print(f"STARK model loaded: {hef_path}")
        for info in self.input_infos:
            print(f"  Input:  {info.name} shape={tuple(info.shape)}")
        for info in self.output_infos:
            print(f"  Output: {info.name} shape={tuple(info.shape)}")

    @staticmethod
    def _frame_shape(expected_shape):
        expected = tuple(int(v) for v in expected_shape)
        if len(expected) > 1 and expected[0] == 1:
            return expected[1:]
        return expected

    @classmethod
    def _as_single_frame(cls, array, expected_shape):
        """Return a contiguous (1, ...) buffer for one image.

        InferVStreams counts frames from shape[0] for every input; both
        STARK inputs must report exactly one frame.
        """
        frame_shape = cls._frame_shape(expected_shape)
        expected_items = int(np.prod(frame_shape))
        if array.size != expected_items:
            raise ValueError(
                f"Buffer size {array.size} != HEF frame size {expected_items} "
                f"(shape {frame_shape})"
            )
        return np.ascontiguousarray(array.reshape((1,) + frame_shape))

    def run(self, template, search):
        """Run one tracking step. Both arrays are RGB 0-255."""
        template = self._as_single_frame(
            np.asarray(template, dtype=np.float32), self._template_shape
        )
        search = self._as_single_frame(
            np.asarray(search, dtype=np.float32), self._search_shape
        )
        try:
            return self._pipe.infer({
                self.template_input.name: template,
                self.search_input.name: search,
            })
        except Exception as e:
            print(f"Hailo inference error: {e}")
            return None

    def release(self):
        try:
            self._pipe_ctx.__exit__(None, None, None)
        except Exception:
            pass
        try:
            self._act.__exit__(None, None, None)
        except Exception:
            pass
        try:
            self.target.release()
        except Exception:
            pass
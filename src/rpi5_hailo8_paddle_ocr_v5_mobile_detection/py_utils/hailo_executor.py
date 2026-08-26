import numpy as np
from hailo_platform import ConfigureParams, FormatType, HailoStreamInterface, HEF, InferVStreams, InputVStreamParams, OutputVStreamParams, VDevice


class HailoInfer:
    def __init__(self, hef_path):
        self.hef = HEF(hef_path)
        self.target = VDevice()
        params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.target.configure(self.hef, params)[0]
        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.output_infos = self.hef.get_output_vstream_infos()
        self.input_h, self.input_w, _ = self.input_info.shape
        inputs = InputVStreamParams.make(self.network_group, format_type=FormatType.UINT8)
        outputs = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        self.activation = self.network_group.activate(self.network_group.create_params())
        self.activation.__enter__()
        self.streams = InferVStreams(self.network_group, inputs, outputs)
        self.pipe = self.streams.__enter__()

    def run(self, image):
        if image.ndim == 3:
            image = image[None, ...]
        return self.pipe.infer({self.input_info.name: image.astype(np.uint8, copy=False)})

    def release(self):
        self.streams.__exit__(None, None, None)
        self.activation.__exit__(None, None, None)
        self.target.release()

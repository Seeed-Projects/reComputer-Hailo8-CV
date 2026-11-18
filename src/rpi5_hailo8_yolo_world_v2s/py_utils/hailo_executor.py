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
    """HailoRT 4.23 session for the two-input YOLO-World HEF.

    The image input is supplied as float32 (0-255) and HailoRT converts it to
    the HEF's native UINT8 input; the CLIP text embedding input is float32.
    Both inputs are passed to every inference call.
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
        self.image_input = None
        self.text_input = None
        for info in self.input_infos:
            shape = tuple(int(v) for v in info.shape)
            if len(shape) >= 3 and shape[-1] == 3 and shape[-2] > 32:
                self.image_input = info
            elif 512 in shape:
                self.text_input = info
        if self.image_input is None:
            raise ValueError(
                "No image input (H, W, 3) found in HEF: "
                + ", ".join(f"{i.name}{tuple(i.shape)}" for i in self.input_infos)
            )
        if self.text_input is None:
            raise ValueError(
                "YOLO-World HEF has no 512-d text-embedding input: "
                + ", ".join(f"{i.name}{tuple(i.shape)}" for i in self.input_infos)
            )

        image_shape = tuple(int(v) for v in self.image_input.shape)
        if len(image_shape) == 4:
            self.input_h, self.input_w = image_shape[1], image_shape[2]
        else:
            self.input_h, self.input_w = image_shape[0], image_shape[1]
        self._image_shape = image_shape
        self._text_shape = tuple(int(v) for v in self.text_input.shape)
        self._text_embeddings = None

        self.output_infos = self.hef.get_output_vstream_infos()

        # Keep every input in its native HEF format: UINT8 for the image and
        # FLOAT32 for the CLIP embeddings. AUTO lets each vstream use the
        # device format instead of forcing one dtype across both inputs.
        in_p = InputVStreamParams.make(self.network_group)
        out_p = OutputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32
        )

        # Hold activation + pipeline for the lifetime of the wrapper.
        # Rebuilding them per-frame collapses FPS to single digits.
        self._act = self.network_group.activate(self.ng_params)
        self._act.__enter__()
        self._pipe_ctx = InferVStreams(self.network_group, in_p, out_p)
        self._pipe = self._pipe_ctx.__enter__()

        print(f"YOLO-World model loaded: {hef_path}")
        for info in self.input_infos:
            print(f"  Input:  {info.name} shape={tuple(info.shape)}")
        for info in self.output_infos:
            print(f"  Output: {info.name} shape={tuple(info.shape)}")

    def set_text_embeddings(self, embeddings):
        """Install the vocabulary embeddings used by every inference."""
        arr = np.ascontiguousarray(embeddings, dtype=np.float32)
        expected = int(np.prod(self._text_shape))
        if arr.size != expected:
            raise ValueError(
                f"Text embedding size {arr.size} != HEF input size {expected} "
                f"(shape {self._text_shape})"
            )
        self._text_embeddings = arr.reshape(self._text_shape)
        print(
            f"Text embeddings installed: shape={self._text_embeddings.shape}, "
            f"rows={self._text_embeddings.size // 512}",
            flush=True,
        )

    @staticmethod
    def _frame_shape(expected_shape):
        """Return the per-frame shape, dropping a leading batch dim of 1."""
        expected = tuple(int(v) for v in expected_shape)
        if len(expected) > 1 and expected[0] == 1:
            return expected[1:]
        return expected

    @classmethod
    def _as_single_frame(cls, array, expected_shape):
        """Reshape one sample to (1, ...) so HailoRT sees one frame per input.

        InferVStreams counts frames from shape[0] for every input. The image
        vstream is (H, W, C) while the text vstream is (1, 80, 512); without a
        common leading batch dim HailoRT compares 640 with 1 and rejects the
        call.
        """
        frame_shape = cls._frame_shape(expected_shape)
        expected_items = int(np.prod(frame_shape))
        if array.size != expected_items:
            raise ValueError(
                f"Buffer size {array.size} != HEF frame size {expected_items} "
                f"(shape {frame_shape})"
            )
        return np.ascontiguousarray(array.reshape((1,) + frame_shape))

    def run(self, image, text_embeddings=None):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        image = self._as_single_frame(image, self._image_shape)

        if text_embeddings is None:
            embeddings = self._text_embeddings
        else:
            embeddings = np.ascontiguousarray(text_embeddings, dtype=np.float32)
            embeddings = self._as_single_frame(embeddings, self._text_shape)
        if embeddings is None:
            raise RuntimeError(
                "Text embeddings are not installed; call set_text_embeddings()"
            )
        embeddings = np.ascontiguousarray(embeddings)

        try:
            return self._pipe.infer({
                self.image_input.name: image,
                self.text_input.name: embeddings,
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
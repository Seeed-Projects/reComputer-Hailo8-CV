import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import web_detection as wd  # noqa: E402


class FakeDetector:
    def detect(self, frame, score_threshold, nms_threshold):
        row = np.zeros(15, dtype=np.float32)
        row[:4] = [40.0, 50.0, 80.0, 80.0]
        row[14] = 0.95
        return row.reshape(1, 15)


class FakeModel:
    def run(self, image):
        if image.shape != (192, 192, 3):
            raise AssertionError(image.shape)
        landmarks = np.zeros((1, 1, 1404), dtype=np.float32)
        landmarks[..., 0::3] = 96.0
        landmarks[..., 1::3] = 96.0
        return {
            "face_landmarks_lite/output_layer1": landmarks,
            "face_landmarks_lite/output_layer2": np.array(
                [[[0.9]]], dtype=np.float32
            ),
        }


class PipelineSyntheticTests(unittest.TestCase):
    def test_process_frame_returns_468_mapped_points(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        annotated, results = wd.process_frame(
            frame, FakeModel(), FakeDetector(), 0.5, 0.3, 4
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]["points"]), 468)
        self.assertAlmostEqual(float(results[0]["points"][0][0]), 80.0, 1)
        self.assertAlmostEqual(float(results[0]["points"][0][1]), 90.0, 1)
        self.assertGreater(int(np.count_nonzero(annotated)), 0)


if __name__ == "__main__":
    unittest.main()
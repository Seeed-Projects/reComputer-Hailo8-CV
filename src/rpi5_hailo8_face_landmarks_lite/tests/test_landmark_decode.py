import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from py_utils.landmark_utils import (  # noqa: E402
    FACE_LANDMARK_COUNT,
    FACE_MODEL_SIZE,
    expand_face_box,
    extract_landmarks,
    map_landmarks_to_frame,
)


class LandmarkDecodeTests(unittest.TestCase):
    def test_pixel_coordinates_are_kept(self):
        raw = np.linspace(0, FACE_MODEL_SIZE, FACE_LANDMARK_COUNT * 3)
        outputs = {
            "face_landmarks_lite/output_layer1": raw.reshape(1, 1, 1404),
            "face_landmarks_lite/output_layer2": np.array([[[0.9]]]),
        }
        points, score, normalized = extract_landmarks(outputs)
        self.assertEqual(points.shape, (468, 3))
        self.assertFalse(normalized)
        self.assertAlmostEqual(score, 0.9, places=5)
        self.assertAlmostEqual(float(points[0, 0]), 0.0, places=4)
        self.assertAlmostEqual(
            float(points[-1, 2]), FACE_MODEL_SIZE, places=4
        )

    def test_normalized_coordinates_are_scaled(self):
        raw = np.zeros(FACE_LANDMARK_COUNT * 3, dtype=np.float32)
        raw[0::3] = 0.5
        raw[1::3] = 0.5
        outputs = {
            "face_landmarks_lite/output_layer1": raw.reshape(1404),
            "face_landmarks_lite/output_layer2": np.array([0.0]),
        }
        points, score, normalized = extract_landmarks(outputs)
        self.assertTrue(normalized)
        self.assertAlmostEqual(score, 0.0, places=5)
        self.assertAlmostEqual(
            float(np.max(points[:, :2])), FACE_MODEL_SIZE / 2.0, places=4
        )

    def test_missing_landmarks_returns_none(self):
        points, score, normalized = extract_landmarks(
            {"only_score": np.array([0.5])}
        )
        self.assertIsNone(points)
        self.assertIsNone(score)
        self.assertFalse(normalized)

    def test_face_box_is_square_and_clamped(self):
        box = expand_face_box([10, 20, 40, 60], 100, 100, margin=0.2)
        self.assertIsNotNone(box)
        x1, y1, x2, y2 = box
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, 100)
        self.assertLessEqual(y2, 100)
        self.assertGreater(x2, x1)
        self.assertGreater(y2, y1)

    def test_landmarks_map_back_to_crop(self):
        points = np.zeros((468, 3), dtype=np.float32)
        points[:, 0] = 96.0
        points[:, 1] = 96.0
        mapped = map_landmarks_to_frame(points, (100, 200, 300, 400))
        self.assertAlmostEqual(float(mapped[0, 0]), 200.0, places=4)
        self.assertAlmostEqual(float(mapped[0, 1]), 300.0, places=4)


if __name__ == "__main__":
    unittest.main()
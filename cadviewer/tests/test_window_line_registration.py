"""Tests for window line registration image detectors."""

import unittest

import numpy as np

from cadviewer.registration.window_line_registration import _detect_registration_geometry


class WindowLineRegistrationTest(unittest.TestCase):
    def test_bright_mode_detects_backlit_window_component(self) -> None:
        gray = np.full((700, 900), 80, dtype=np.uint8)
        gray[180:520, 220:700] = 245

        side_positions, bbox, confidence, _side_lines, corners = (
            _detect_registration_geometry(
                gray,
                target_aspect=480.0 / 340.0,
                detection_mode="bright",
            )
        )

        self.assertEqual(bbox, (220, 180, 699, 519))
        self.assertGreater(confidence, 0.95)
        self.assertAlmostEqual(side_positions["left"], 220.0, delta=1.0)
        self.assertAlmostEqual(side_positions["right"], 699.0, delta=1.0)
        self.assertAlmostEqual(side_positions["top"], 180.0, delta=1.0)
        self.assertAlmostEqual(side_positions["bottom"], 519.0, delta=1.0)
        self.assertEqual(corners.shape, (4, 2))

    def test_bright_mode_accepts_reciprocal_aspect_when_target_is_known(self) -> None:
        gray = np.full((1200, 900), 60, dtype=np.uint8)
        gray[120:1020, 250:520] = 245

        side_positions, bbox, confidence, _side_lines, corners = (
            _detect_registration_geometry(
                gray,
                target_aspect=270.0 / 900.0,
                detection_mode="bright",
            )
        )

        self.assertEqual(bbox, (250, 120, 519, 1019))
        self.assertGreater(confidence, 0.95)
        self.assertAlmostEqual(side_positions["left"], 250.0, delta=1.0)
        self.assertAlmostEqual(side_positions["right"], 519.0, delta=1.0)
        self.assertAlmostEqual(side_positions["top"], 120.0, delta=1.0)
        self.assertAlmostEqual(side_positions["bottom"], 1019.0, delta=1.0)
        self.assertEqual(corners.shape, (4, 2))


if __name__ == "__main__":
    unittest.main()

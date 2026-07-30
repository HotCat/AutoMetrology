"""Tests for the backlight hollow-window fitter used by dual-light metrology."""

from __future__ import annotations

import unittest

import numpy as np

from tools.validate_backlight_hollow_window import _select_bright_window_component


class BacklightHollowWindowTest(unittest.TestCase):
    def test_selects_tall_backlight_window_with_known_cad_aspect(self) -> None:
        gray = np.full((1600, 1400), 8, dtype=np.uint8)
        gray[250:1350, 600:980] = 245

        _component, bbox, threshold, score = _select_bright_window_component(
            gray,
            target_aspect=1100.0 / 380.0,
        )

        self.assertEqual(bbox, (600, 250, 979, 1349))
        self.assertGreaterEqual(threshold, 150)
        self.assertTrue(np.isfinite(score))


if __name__ == "__main__":
    unittest.main()

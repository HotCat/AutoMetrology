"""Tests for measurement pipeline pair-fitting helpers."""

import unittest

from cadviewer.measurement.measurement_pipeline import MeasurementPipeline


class MeasurementPipelineTest(unittest.TestCase):
    def test_parallel_line_gap_uses_cad_geometry(self) -> None:
        left_outer = {
            "x1": 2586.611758333621,
            "y1": 1294.809387634002,
            "x2": 2586.611758333621,
            "y2": 1191.209387634002,
        }
        left_window = {
            "x1": 2597.611758333622,
            "y1": 1206.924587614001,
            "x2": 2597.611758333622,
            "y2": 1279.104587614001,
        }

        gap = MeasurementPipeline._parallel_line_gap_mm(left_outer, left_window)

        self.assertIsNotNone(gap)
        self.assertAlmostEqual(gap, 11.0, places=6)


if __name__ == "__main__":
    unittest.main()

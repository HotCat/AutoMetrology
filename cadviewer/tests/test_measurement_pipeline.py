"""Tests for measurement pipeline pair-fitting helpers."""

import unittest

import numpy as np

from cadviewer.models.feature import FeatureType
from cadviewer.models.measured_feature import MeasuredFeature
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

    def test_line_geometries_parallel_rejects_perpendicular(self) -> None:
        horizontal = {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 0.0}
        vertical = {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 10.0}

        self.assertFalse(
            MeasurementPipeline._line_geometries_parallel(horizontal, vertical)
        )

    def test_line_distance_between_measured_uses_fitted_world_geometry(self) -> None:
        a = self._line_measurement("a", 0.0)
        b = self._line_measurement("b", 7.25)

        distance = MeasurementPipeline._line_distance_between_measured(a, b)

        self.assertAlmostEqual(distance, 7.25, places=6)

    def test_center_between_line_fits_averages_pixel_and_world_geometry(self) -> None:
        pipeline = MeasurementPipeline.__new__(MeasurementPipeline)
        feature = type("Feature", (), {
            "feature_id": "cad-line",
            "feature_type": FeatureType.LINE,
        })()
        side_a = self._line_measurement("cad-line", 10.0)
        side_b = self._line_measurement("cad-line", 14.0)

        centered = pipeline._center_between_line_fits(feature, side_a, side_b)

        self.assertIsNotNone(centered)
        self.assertEqual(centered.cad_feature_id, "cad-line")
        self.assertEqual(centered.detection_method, "perpendicular_scanline_stroke_center")
        self.assertAlmostEqual(centered.fitted_geometry_world["y1"], 12.0)
        self.assertAlmostEqual(centered.fitted_geometry_world["y2"], 12.0)
        self.assertEqual(len(centered.edge_points), 4)

    @staticmethod
    def _line_measurement(cad_id: str, y: float) -> MeasuredFeature:
        return MeasuredFeature(
            feature_id=f"meas-{cad_id}-{y}",
            cad_feature_id=cad_id,
            feature_type=FeatureType.LINE,
            fitted_geometry={"x1": 0.0, "y1": y, "x2": 10.0, "y2": y},
            fitted_geometry_world={"x1": 0.0, "y1": y, "x2": 10.0, "y2": y},
            edge_points=np.array([[0.0, y], [10.0, y]], dtype=np.float64),
            roi_bbox=(0, 0, 10, 20),
            residual_error=0.1,
            confidence=0.9,
            detection_method="test",
            source_type="FITTED",
        )


if __name__ == "__main__":
    unittest.main()

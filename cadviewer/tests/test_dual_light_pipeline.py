"""Tests for dual-light pipeline safety diagnostics."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from cadviewer.measurement import dual_light_pipeline
from cadviewer.models.query import QueryResult


class DualLightPipelineDiagnosticsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = object.__new__(dual_light_pipeline.DualLightMeasurementPipeline)
        self.pipeline._repo = object()
        self.pipeline._ring_bgr = np.zeros((20, 30, 3), dtype=np.uint8)
        self.pipeline._window_features = [
            SimpleNamespace(feature_id=f"line-{idx}") for idx in range(4)
        ]
        self.pipeline._pixel_size_mm = 0.01
        self.pipeline._image_corners = np.array(
            [
                [10.0, 10.0],
                [20.0, 10.0],
                [20.0, 18.0],
                [10.0, 18.0],
            ],
            dtype=np.float64,
        )
        self._original_register = dual_light_pipeline.register_window_lines

    def tearDown(self) -> None:
        dual_light_pipeline.register_window_lines = self._original_register

    def test_dark_ring_detector_failure_is_diagnostic_only(self) -> None:
        def fail_detector(*args, **kwargs):
            raise RuntimeError("No suitable dark window component detected")

        dual_light_pipeline.register_window_lines = fail_detector

        diagnostic = self.pipeline.validate_ring_pose_consistency()

        self.assertEqual(diagnostic["status"], "unavailable")
        self.assertIn("No suitable dark window component detected", diagnostic["error"])

    def test_successful_ring_pose_mismatch_still_rejects_measurement(self) -> None:
        def mismatched_detector(*args, **kwargs):
            return SimpleNamespace(
                image_corners=self.pipeline._image_corners + 100.0,
                confidence=1.0,
                component_bbox=(0, 0, 10, 10),
            )

        dual_light_pipeline.register_window_lines = mismatched_detector

        with self.assertRaisesRegex(RuntimeError, "window poses disagree"):
            self.pipeline.validate_ring_pose_consistency(max_mean_corner_delta_px=25.0)

    def test_run_dual_light_measurement_rejects_ambiguous_orientation(self) -> None:
        class FakePipeline:
            def __init__(self, *args, **kwargs):
                self.registration_debug = {}
                self.measurement_transform = np.eye(3, dtype=np.float64)

            def validate_ring_pose_consistency(self):
                return {"status": "unavailable"}

            def validate_orientation_with_ring_prints(self, *args, **kwargs):
                return {
                    "status": "ambiguous_180_warning",
                    "warning": "ambiguous orientation",
                    "candidates": [],
                }

            def save_artifacts(self, *args, **kwargs):
                raise AssertionError("save_artifacts should not run")

            def get_debug_data(self):
                return {}

            def geometric_consistency_diagnostics(self, results):
                return {}

        class GuardEvaluator:
            def __init__(self, *args, **kwargs):
                raise AssertionError("QueryEvaluator should not run")

        with patch.object(dual_light_pipeline, "DualLightMeasurementPipeline", FakePipeline), \
             patch.object(dual_light_pipeline, "QueryEvaluator", GuardEvaluator):
            with self.assertRaisesRegex(RuntimeError, "orientation could not be resolved"):
                dual_light_pipeline.run_dual_light_measurement(
                    repo=object(),
                    query_text="lines(AC5C:7, AB81:7)",
                    backlight_image=np.zeros((10, 10, 3), dtype=np.uint8),
                    ring_light_image=np.zeros((10, 10, 3), dtype=np.uint8),
                    edge_tokens=["a", "b", "c", "d"],
                    pixel_size_mm=0.01,
                )

    def test_run_dual_light_measurement_allows_ambiguous_orientation_when_guard_disabled(self) -> None:
        class FakePipeline:
            def __init__(self, *args, **kwargs):
                self.registration_debug = {}
                self.measurement_transform = np.eye(3, dtype=np.float64)

            def validate_ring_pose_consistency(self):
                raise AssertionError("ring pose guard should be skipped")

            def validate_orientation_with_ring_prints(self, *args, **kwargs):
                raise AssertionError("orientation guard should be skipped")

            def get_debug_data(self):
                return {}

            def geometric_consistency_diagnostics(self, results):
                return {}

        class FakeEvaluator:
            def __init__(self, *args, **kwargs):
                pass

            def evaluate(self, query_text):
                return [QueryResult(status="ok")]

        with patch.object(dual_light_pipeline, "DualLightMeasurementPipeline", FakePipeline), \
             patch.object(dual_light_pipeline, "QueryEvaluator", FakeEvaluator):
            result = dual_light_pipeline.run_dual_light_measurement(
                repo=object(),
                query_text="lines(AC5C:7, AB81:7)",
                backlight_image=np.zeros((10, 10, 3), dtype=np.uint8),
                ring_light_image=np.zeros((10, 10, 3), dtype=np.uint8),
                edge_tokens=["a", "b", "c", "d"],
                pixel_size_mm=0.01,
                orientation_guard_enabled=False,
            )

        self.assertEqual(len(result.results), 1)
        self.assertFalse(result.registration["orientation_guard_enabled"])
        self.assertEqual(
            result.registration["orientation_validation"]["status"],
            "skipped",
        )
        self.assertEqual(result.registration["ring_pose_consistency"]["status"], "skipped")


if __name__ == "__main__":
    unittest.main()

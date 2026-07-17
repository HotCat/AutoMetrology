"""Tests for stable hollow-window dimensions from opposite fitted lines."""

import math
import unittest

import numpy as np

from cadviewer.measurement.window_dimensions import (
    corner_segment_distances,
    measure_window_line_dimensions,
    side_line_corners,
)


class WindowDimensionsTest(unittest.TestCase):
    def test_axis_aligned_perfect_rectangle(self) -> None:
        lines = self._rectangle_lines(100.0, 200.0, 300.0, 160.0, 0.0)

        dims = measure_window_line_dimensions(lines, pixel_size_mm=0.02)

        self.assertAlmostEqual(dims.opposite_line_width_px, 300.0, places=6)
        self.assertAlmostEqual(dims.opposite_line_height_px, 160.0, places=6)
        self.assertAlmostEqual(dims.opposite_line_width_mm, 6.0, places=6)
        self.assertAlmostEqual(dims.opposite_line_height_mm, 3.2, places=6)

    def test_rectangle_translation_invariance(self) -> None:
        base = measure_window_line_dimensions(
            self._rectangle_lines(100.0, 200.0, 300.0, 160.0, 0.0),
            pixel_size_mm=0.01,
        )
        shifted = measure_window_line_dimensions(
            self._rectangle_lines(1200.0, -700.0, 300.0, 160.0, 0.0),
            pixel_size_mm=0.01,
        )

        self.assertAlmostEqual(base.opposite_line_width_px, shifted.opposite_line_width_px)
        self.assertAlmostEqual(base.opposite_line_height_px, shifted.opposite_line_height_px)

    def test_rotated_rectangle(self) -> None:
        lines = self._rectangle_lines(100.0, 200.0, 300.0, 160.0, math.radians(7.0))

        dims = measure_window_line_dimensions(lines, pixel_size_mm=0.02)

        self.assertAlmostEqual(dims.opposite_line_width_px, 300.0, places=6)
        self.assertAlmostEqual(dims.opposite_line_height_px, 160.0, places=6)

    def test_small_independent_angle_perturbations_report_quality(self) -> None:
        lines = self._rectangle_lines(100.0, 200.0, 300.0, 160.0, 0.0)
        lines["right"] = self._line_from_normal_and_point(
            self._unit([math.cos(math.radians(0.08)), math.sin(math.radians(0.08))]),
            np.array([250.0, 200.0]),
        )

        dims = measure_window_line_dimensions(lines, pixel_size_mm=0.01)

        self.assertGreater(dims.width_stats.angle_difference_deg, 0.05)
        self.assertAlmostEqual(dims.opposite_line_width_px, 300.0, places=3)

    def test_corner_intersection_jitter_affects_old_dimension_more(self) -> None:
        base = self._rectangle_lines(0.0, 0.0, 300.0, 160.0, 0.0)
        jittered = dict(base)
        jittered["top"] = self._line_from_normal_and_point(
            self._unit([math.cos(math.radians(-89.0)), math.sin(math.radians(-89.0))]),
            np.array([0.0, -80.0]),
        )
        jittered["bottom"] = self._line_from_normal_and_point(
            self._unit([math.cos(math.radians(89.5)), math.sin(math.radians(89.5))]),
            np.array([0.0, 80.0]),
        )

        base_dims = measure_window_line_dimensions(base, pixel_size_mm=1.0)
        jitter_dims = measure_window_line_dimensions(jittered, pixel_size_mm=1.0)
        old_base = corner_segment_distances(side_line_corners(base))["corner_segment_width_px"]
        old_jitter = corner_segment_distances(side_line_corners(jittered))["corner_segment_width_px"]

        new_delta = abs(jitter_dims.opposite_line_width_px - base_dims.opposite_line_width_px)
        old_delta = abs(old_jitter - old_base)
        self.assertLess(new_delta, old_delta)
        self.assertLess(new_delta, 0.05)

    def test_known_left_right_normal_offsets(self) -> None:
        lines = self._rectangle_lines(0.0, 0.0, 300.0, 160.0, math.radians(12.0))
        lines["left"] = self._offset_line(lines["left"], 5.0)
        lines["right"] = self._offset_line(lines["right"], 7.0)

        dims = measure_window_line_dimensions(lines, pixel_size_mm=1.0)

        self.assertAlmostEqual(dims.opposite_line_width_px, 312.0, places=6)

    def test_known_top_bottom_normal_offsets(self) -> None:
        lines = self._rectangle_lines(0.0, 0.0, 300.0, 160.0, math.radians(-9.0))
        lines["top"] = self._offset_line(lines["top"], 3.0)
        lines["bottom"] = self._offset_line(lines["bottom"], 4.0)

        dims = measure_window_line_dimensions(lines, pixel_size_mm=1.0)

        self.assertAlmostEqual(dims.opposite_line_height_px, 167.0, places=6)

    def test_rigid_rotation_invariance(self) -> None:
        widths = []
        heights = []
        for angle in (0.0, 3.0, 17.0, 45.0, 89.0):
            dims = measure_window_line_dimensions(
                self._rectangle_lines(50.0, -30.0, 300.0, 160.0, math.radians(angle)),
                pixel_size_mm=0.01,
            )
            widths.append(dims.opposite_line_width_px)
            heights.append(dims.opposite_line_height_px)

        self.assertLess(max(widths) - min(widths), 1e-9)
        self.assertLess(max(heights) - min(heights), 1e-9)

    def test_pixel_size_conversion(self) -> None:
        dims = measure_window_line_dimensions(
            self._rectangle_lines(0.0, 0.0, 300.0, 160.0, 0.0),
            pixel_size_mm=0.025,
        )

        self.assertAlmostEqual(dims.opposite_line_width_mm, 7.5, places=6)
        self.assertAlmostEqual(dims.opposite_line_height_mm, 4.0, places=6)

    def test_point_corrector_uses_one_measurement_coordinate_space(self) -> None:
        def corrector(points: np.ndarray) -> np.ndarray:
            corrected = np.asarray(points, dtype=np.float64).copy()
            corrected[:, 0] *= 1.01
            corrected[:, 1] *= 0.99
            return corrected

        dims = measure_window_line_dimensions(
            self._rectangle_lines(0.0, 0.0, 300.0, 160.0, 0.0),
            pixel_size_mm=0.01,
            point_corrector=corrector,
        )

        self.assertAlmostEqual(dims.opposite_line_width_px, 303.0, places=6)
        self.assertAlmostEqual(dims.opposite_line_height_px, 158.4, places=6)
        self.assertAlmostEqual(dims.opposite_line_width_mm, 3.03, places=6)
        self.assertAlmostEqual(dims.opposite_line_height_mm, 1.584, places=6)

    def test_cad_nominal_size_does_not_participate(self) -> None:
        lines = self._rectangle_lines(0.0, 0.0, 300.0, 160.0, 0.0)

        first = measure_window_line_dimensions(lines, pixel_size_mm=0.01)
        second = measure_window_line_dimensions(lines, pixel_size_mm=0.01)

        self.assertEqual(first.opposite_line_width_px, second.opposite_line_width_px)
        self.assertEqual(first.opposite_line_height_px, second.opposite_line_height_px)

    def test_intensity_bias_simulation_keeps_opposite_line_stable(self) -> None:
        base = self._rectangle_lines(0.0, 0.0, 300.0, 160.0, 0.0)
        widths = []
        old_widths = []
        for angle_deg in (-0.15, -0.05, 0.0, 0.05, 0.15):
            lines = dict(base)
            lines["top"] = self._line_from_normal_and_point(
                self._unit([
                    math.cos(math.radians(-90.0 + angle_deg)),
                    math.sin(math.radians(-90.0 + angle_deg)),
                ]),
                np.array([0.0, -80.0]),
            )
            lines["bottom"] = self._line_from_normal_and_point(
                self._unit([
                    math.cos(math.radians(90.0 - angle_deg * 0.6)),
                    math.sin(math.radians(90.0 - angle_deg * 0.6)),
                ]),
                np.array([0.0, 80.0]),
            )
            dims = measure_window_line_dimensions(lines, pixel_size_mm=1.0)
            widths.append(dims.opposite_line_width_px)
            old_widths.append(
                corner_segment_distances(side_line_corners(lines))["corner_segment_width_px"]
            )

        self.assertLess(max(widths) - min(widths), 0.05)
        self.assertGreater(max(old_widths) - min(old_widths), max(widths) - min(widths))

    @staticmethod
    def _rectangle_lines(
        cx: float,
        cy: float,
        width: float,
        height: float,
        theta: float,
    ) -> dict[str, tuple[float, float, float]]:
        ux = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
        uy = np.array([-math.sin(theta), math.cos(theta)], dtype=np.float64)
        center = np.array([cx, cy], dtype=np.float64)
        return {
            "left": WindowDimensionsTest._line_from_normal_and_point(-ux, center - ux * width * 0.5),
            "right": WindowDimensionsTest._line_from_normal_and_point(ux, center + ux * width * 0.5),
            "top": WindowDimensionsTest._line_from_normal_and_point(-uy, center - uy * height * 0.5),
            "bottom": WindowDimensionsTest._line_from_normal_and_point(uy, center + uy * height * 0.5),
        }

    @staticmethod
    def _line_from_normal_and_point(
        normal: np.ndarray,
        point: np.ndarray,
    ) -> tuple[float, float, float]:
        normal = WindowDimensionsTest._unit(normal)
        c = -float(normal @ point)
        return float(normal[0]), float(normal[1]), c

    @staticmethod
    def _offset_line(
        line: tuple[float, float, float],
        outward_px: float,
    ) -> tuple[float, float, float]:
        a, b, c = line
        return a, b, c - outward_px

    @staticmethod
    def _unit(values) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        return arr / np.linalg.norm(arr)


if __name__ == "__main__":
    unittest.main()

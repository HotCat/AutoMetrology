"""
Local feature fitting for precision metrology.

After global registration, individual CAD features are transformed to
image space, local ROIs are predicted, and subpixel fitting is performed
for circles, lines, and other geometric primitives.

This module implements the measurement stage that happens AFTER
registration is frozen. It does NOT optimize the global transform.
"""

from __future__ import annotations

import math
import numpy as np

from ..registration import affine_solver
from ..registration.image_extractor import ImageFeatureExtractor


_MIN_CIRCLE_PTS = 10
_MIN_LINE_PTS = 6
_MAX_CIRCLE_RESIDUAL = 3.0
_MAX_LINE_RESIDUAL = 3.0
_ROI_CIRCLE_PADDING = 15
_ROI_LINE_PADDING = 15


class LocalROIPredictor:
    """Predict local ROIs from transformed CAD features."""

    def __init__(
        self, affine: np.ndarray, pixel_size_mm: float,
    ) -> None:
        """
        Args:
            affine: 3x3 matrix mapping pixel → CAD world
            pixel_size_mm: mm per pixel
        """
        self._affine = affine
        self._inv_affine = (
            affine_solver.invert(affine) if affine is not None else None
        )
        self._pixel_size_mm = pixel_size_mm

    def predict_circle_roi(
        self, geometry: dict, padding: float = _ROI_CIRCLE_PADDING,
    ) -> tuple[tuple[int, int, int, int], np.ndarray, float] | None:
        """Predict ROI for a circle feature.

        Args:
            geometry: dict with 'cx', 'cy', 'radius'
            padding: extra pixels around predicted circle

        Returns:
            (roi, pixel_center, pixel_radius) or None
        """
        if self._inv_affine is None:
            return None

        cx, cy = geometry.get("cx", 0), geometry.get("cy", 0)
        r = geometry.get("radius", 1.0)

        pixel_center = affine_solver.apply(
            self._inv_affine, np.array([[cx, cy]]),
        )[0]

        offset_pt = affine_solver.apply(
            self._inv_affine, np.array([[cx + r, cy]]),
        )[0]
        pixel_radius = abs(offset_pt[0] - pixel_center[0])
        if pixel_radius < 5:
            pixel_radius = 30

        roi = (
            int(pixel_center[0] - pixel_radius - padding),
            int(pixel_center[1] - pixel_radius - padding),
            int(pixel_center[0] + pixel_radius + padding),
            int(pixel_center[1] + pixel_radius + padding),
        )
        return roi, pixel_center, pixel_radius

    def predict_line_roi(
        self, geometry: dict, padding: float = _ROI_LINE_PADDING,
    ) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray] | None:
        """Predict ROI for a line feature.

        Args:
            geometry: dict with 'x1', 'y1', 'x2', 'y2'
            padding: extra pixels around predicted line

        Returns:
            (roi, pixel_p1, pixel_p2) or None
        """
        if self._inv_affine is None:
            return None

        x1, y1 = geometry.get("x1", 0), geometry.get("y1", 0)
        x2, y2 = geometry.get("x2", 0), geometry.get("y2", 0)
        pixel_pts = affine_solver.apply(
            self._inv_affine, np.array([[x1, y1], [x2, y2]]),
        )

        px_min = min(pixel_pts[0, 0], pixel_pts[1, 0])
        px_max = max(pixel_pts[0, 0], pixel_pts[1, 0])
        py_min = min(pixel_pts[0, 1], pixel_pts[1, 1])
        py_max = max(pixel_pts[0, 1], pixel_pts[1, 1])

        line_len = math.sqrt(
            (px_max - px_min) ** 2 + (py_max - py_min) ** 2,
        )
        pad = max(padding, line_len * 0.15)

        roi = (
            int(px_min - pad), int(py_min - pad),
            int(px_max + pad), int(py_max + pad),
        )
        return roi, pixel_pts[0], pixel_pts[1]


class CircleMeasurementEngine:
    """Subpixel circle fitting for local metrology."""

    def __init__(self, edge_cache: np.ndarray, affine: np.ndarray) -> None:
        """
        Args:
            edge_cache: Nx2 float64 full-image Canny edge points (pixel coords)
            affine: 3x3 matrix pixel → world
        """
        self._edges = edge_cache
        self._affine = affine

    def fit(
        self,
        geometry: dict,
        roi_predictor: LocalROIPredictor,
    ) -> tuple[dict, float, np.ndarray] | None:
        """Fit a circle in the predicted ROI.

        Returns:
            (fitted_dict, residual, world_center) or None
        """
        pred = roi_predictor.predict_circle_roi(geometry)
        if pred is None:
            return None
        roi, _, _ = pred

        roi_edges = ImageFeatureExtractor.extract_edge_points_in_roi(
            self._edges, roi,
        )
        if len(roi_edges) < _MIN_CIRCLE_PTS:
            return None

        fitted, residual = ImageFeatureExtractor.fit_circle_subpixel(roi_edges)
        if fitted is None or residual > _MAX_CIRCLE_RESIDUAL:
            return None

        pixel_center = np.array([[fitted["cx"], fitted["cy"]]])
        world_center = affine_solver.apply(self._affine, pixel_center)[0]
        return fitted, residual, world_center


class LineMeasurementEngine:
    """Subpixel line fitting for local metrology."""

    def __init__(self, edge_cache: np.ndarray, affine: np.ndarray) -> None:
        self._edges = edge_cache
        self._affine = affine

    def fit(
        self,
        geometry: dict,
        roi_predictor: LocalROIPredictor,
    ) -> tuple[dict, float, np.ndarray] | None:
        """Fit a line in the predicted ROI.

        Returns:
            (fitted_dict, residual, world_pts) or None
        """
        pred = roi_predictor.predict_line_roi(geometry)
        if pred is None:
            return None
        roi, _, _ = pred

        roi_edges = ImageFeatureExtractor.extract_edge_points_in_roi(
            self._edges, roi,
        )
        if len(roi_edges) < _MIN_LINE_PTS:
            return None

        fitted, residual = ImageFeatureExtractor.fit_line_subpixel(roi_edges)
        if fitted is None or residual > _MAX_LINE_RESIDUAL:
            return None

        pixel_pts = np.array([
            [fitted["x1"], fitted["y1"]],
            [fitted["x2"], fitted["y2"]],
        ])
        world_pts = affine_solver.apply(self._affine, pixel_pts)
        return fitted, residual, world_pts

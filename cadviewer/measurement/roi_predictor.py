"""
CAD-guided ROI prediction for local feature measurement.

Projects CAD features into image space via the registration transform
and generates local search regions (ROIs) for edge-based fitting.

The CAD feature is ONLY used to constrain the image search area.
The actual measured geometry comes from image edge data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..registration import affine_solver


@dataclass
class ROIRegion:
    """Axis-aligned bounding box in pixel coordinates."""

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin

    @property
    def center(self) -> tuple[int, int]:
        return (self.xmin + self.xmax) // 2, (self.ymin + self.ymax) // 2

    def clip(self, img_w: int, img_h: int) -> ROIRegion:
        return ROIRegion(
            max(0, self.xmin), max(0, self.ymin),
            min(img_w, self.xmax), min(img_h, self.ymax),
        )


class FeatureROIPredictor:
    """Predict local ROIs from CAD features via registration transform."""

    def __init__(self, affine: np.ndarray) -> None:
        """
        Args:
            affine: 3x3 matrix mapping pixel → CAD world coords
        """
        self._affine = affine
        self._inv_affine = affine_solver.invert(affine)

    def predict_circle_roi(
        self, cad_geometry: dict, padding: float = 15.0,
    ) -> tuple[ROIRegion, np.ndarray, float] | None:
        """Predict ROI for a circle feature.

        Args:
            cad_geometry: dict with 'cx', 'cy', 'radius' (CAD world mm)
            padding: extra pixels around predicted circle

        Returns:
            (ROIRegion, pixel_center, pixel_radius) or None
        """
        cx = cad_geometry.get("cx", 0.0)
        cy = cad_geometry.get("cy", 0.0)
        r = cad_geometry.get("radius", 1.0)

        pixel_center = affine_solver.apply(
            self._inv_affine, np.array([[cx, cy]]),
        )[0]

        offset_pt = affine_solver.apply(
            self._inv_affine, np.array([[cx + r, cy]]),
        )[0]
        pixel_radius = abs(offset_pt[0] - pixel_center[0])
        if pixel_radius < 3:
            pixel_radius = 30.0

        roi = ROIRegion(
            int(pixel_center[0] - pixel_radius - padding),
            int(pixel_center[1] - pixel_radius - padding),
            int(pixel_center[0] + pixel_radius + padding),
            int(pixel_center[1] + pixel_radius + padding),
        )
        return roi, pixel_center, pixel_radius

    def predict_line_roi(
        self, cad_geometry: dict, padding: float = 15.0,
    ) -> tuple[ROIRegion, np.ndarray, np.ndarray] | None:
        """Predict ROI for a line feature.

        Args:
            cad_geometry: dict with 'x1', 'y1', 'x2', 'y2' (CAD world mm)
            padding: extra pixels around predicted line

        Returns:
            (ROIRegion, pixel_p1, pixel_p2) or None
        """
        x1 = cad_geometry.get("x1", 0.0)
        y1 = cad_geometry.get("y1", 0.0)
        x2 = cad_geometry.get("x2", 0.0)
        y2 = cad_geometry.get("y2", 0.0)

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

        roi = ROIRegion(
            int(px_min - pad), int(py_min - pad),
            int(px_max + pad), int(py_max + pad),
        )
        return roi, pixel_pts[0], pixel_pts[1]

    def project_point(self, world_pt: np.ndarray) -> np.ndarray:
        """Project CAD world point to pixel coordinates."""
        return affine_solver.apply(self._inv_affine, world_pt.reshape(1, 2))[0]

    def to_world(self, pixel_pt: np.ndarray) -> np.ndarray:
        """Convert pixel coordinates to CAD world coordinates."""
        return affine_solver.apply(self._affine, pixel_pt.reshape(1, 2))[0]

"""
MeasurementEngine — computes geometric measurements from detected features.

All measurements use image-detected geometry (not CAD nominal),
in world coordinates via the registration affine transform.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..models.image_feature import ImageFeatureRepository
from ..models.correspondence import CorrespondenceMap
from ..models.query import QueryType
from ..registration.affine_solver import apply


class MeasurementEngine:
    """Compute geometric measurements from image-detected features."""

    def __init__(
        self,
        corr_map: CorrespondenceMap,
        image_repo: ImageFeatureRepository,
        affine: np.ndarray,
        pixel_size_mm: float,
    ) -> None:
        self._corr_map = corr_map
        self._image_repo = image_repo
        self._affine = affine
        self._pixel_size_mm = pixel_size_mm

    def measure_circle_distance(
        self, cad_id_1: str, cad_id_2: str
    ) -> Optional[float]:
        """Center-to-center distance between two detected circles."""
        c1 = self._get_world_circle(cad_id_1)
        c2 = self._get_world_circle(cad_id_2)
        if c1 is None or c2 is None:
            return None
        cx1, cy1, _ = c1
        cx2, cy2, _ = c2
        return math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)

    def measure_line_distance(
        self, cad_id_1: str, cad_id_2: str
    ) -> Optional[float]:
        """Perpendicular distance between two fitted lines."""
        l1 = self._get_world_line(cad_id_1)
        l2 = self._get_world_line(cad_id_2)
        if l1 is None or l2 is None:
            return None
        x1, y1, x2, y2 = l1
        # Line 1 direction
        dx1, dy1 = x2 - x1, y2 - y1
        length1 = math.sqrt(dx1 ** 2 + dy1 ** 2)
        if length1 < 1e-12:
            return None
        # Normal of line 1
        nx, ny = -dy1 / length1, dx1 / length1
        # Distance from line 2 endpoints to line 1
        lx1, ly1, lx2, ly2 = l2
        d1 = abs((lx1 - x1) * nx + (ly1 - y1) * ny)
        d2 = abs((lx2 - x1) * nx + (ly2 - y1) * ny)
        return (d1 + d2) / 2

    def compute_nominal(
        self,
        cad_id_1: str,
        cad_id_2: str,
        query_type: QueryType,
        cad_geometry_getter=None,
    ) -> Optional[float]:
        """Compute the same measurement on CAD nominal geometry."""
        if cad_geometry_getter is None:
            return None
        g1 = cad_geometry_getter(cad_id_1)
        g2 = cad_geometry_getter(cad_id_2)
        if g1 is None or g2 is None:
            return None

        if query_type == QueryType.CIRCLE_DISTANCE:
            cx1, cy1 = g1.get("cx", 0), g1.get("cy", 0)
            cx2, cy2 = g2.get("cx", 0), g2.get("cy", 0)
            return math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
        elif query_type == QueryType.LINE_DISTANCE:
            x1, y1 = g1.get("x1", 0), g1.get("y1", 0)
            x2, y2 = g1.get("x2", 0), g1.get("y2", 0)
            dx, dy = x2 - x1, y2 - y1
            length = math.sqrt(dx ** 2 + dy ** 2)
            if length < 1e-12:
                return None
            nx, ny = -dy / length, dx / length
            lx1, ly1 = g2.get("x1", 0), g2.get("y1", 0)
            lx2, ly2 = g2.get("x2", 0), g2.get("y2", 0)
            d1 = abs((lx1 - x1) * nx + (ly1 - y1) * ny)
            d2 = abs((lx2 - x1) * nx + (ly2 - y1) * ny)
            return (d1 + d2) / 2
        return None

    def _get_world_circle(self, cad_id: str) -> Optional[tuple]:
        """Get detected circle center/radius in world coords."""
        corr = self._corr_map.get_for_cad(cad_id)
        if corr is None:
            return None
        img_feat = self._image_repo.get(corr.image_feature_id)
        if img_feat is None:
            return None
        g = img_feat.geometry
        cx = g.get("cx", 0) * self._pixel_size_mm
        cy = g.get("cy", 0) * self._pixel_size_mm
        r = g.get("radius", 0) * self._pixel_size_mm
        return (cx, cy, r)

    def _get_world_line(self, cad_id: str) -> Optional[tuple]:
        """Get detected line endpoints in world coords."""
        corr = self._corr_map.get_for_cad(cad_id)
        if corr is None:
            return None
        img_feat = self._image_repo.get(corr.image_feature_id)
        if img_feat is None:
            return None
        g = img_feat.geometry
        return (
            g.get("x1", 0) * self._pixel_size_mm,
            g.get("y1", 0) * self._pixel_size_mm,
            g.get("x2", 0) * self._pixel_size_mm,
            g.get("y2", 0) * self._pixel_size_mm,
        )

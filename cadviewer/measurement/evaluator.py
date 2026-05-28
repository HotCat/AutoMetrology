"""
QueryEvaluator — orchestrates query execution pipeline.

Parse → Resolve IDs → Measure → Build results.

Supports two modes:
  - CAD-only: measures purely from CAD feature geometry (no image needed)
  - Full: uses image correspondences for measured vs nominal comparison
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from ..models.query import QueryInstruction, QueryResult, QueryType
from ..models.repository import FeatureRepository
from ..registration import affine_solver
from .query_parser import QueryParser

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Min edge points for a reliable subpixel fit
_MIN_CIRCLE_POINTS = 10
_MIN_LINE_POINTS = 6
# Max mean radial residual (pixels) to accept a circle fit
_MAX_CIRCLE_RESIDUAL = 3.0
_MAX_LINE_RESIDUAL = 3.0
# ROI padding in pixels around expected feature location
_ROI_PADDING = 15


class QueryEvaluator:
    """Parse and evaluate measurement queries."""

    def __init__(
        self,
        repo: FeatureRepository,
        image: Optional[np.ndarray] = None,
        affine: Optional[np.ndarray] = None,
    ) -> None:
        self._repo = repo
        self._parser = QueryParser()
        # Grayscale image (uint8, 2D) and pixel→world 3x3 affine
        self._image = image
        self._affine = affine
        # Pre-computed edge point cloud (lazily built once)
        self._edge_cache: Optional[np.ndarray] = None
        # Inverse affine (world → pixel), lazily computed
        self._inv_affine: Optional[np.ndarray] = None
        # Check affine is not identity (no registration done)
        self._has_valid_registration = (
            self._image is not None
            and self._affine is not None
            and not np.allclose(self._affine, np.eye(3), atol=1e-6)
        )

    def evaluate(self, text: str) -> List[QueryResult]:
        """Parse query text and evaluate each instruction."""
        try:
            instructions = self._parser.parse(text)
        except ValueError as e:
            return [QueryResult(
                instruction=None, status="error", error_message=str(e),
            )]

        results = []
        for inst in instructions:
            results.append(self._evaluate_instruction(inst))
        return results

    def _evaluate_instruction(self, inst: QueryInstruction) -> QueryResult:
        """Evaluate a single query instruction."""
        fid1 = self._resolve_id(inst.feature_id_1)
        fid2 = self._resolve_id(inst.feature_id_2)

        if not fid1:
            return QueryResult(
                instruction=inst, status="error",
                error_message=f"Cannot resolve ID: {inst.feature_id_1}",
            )
        if not fid2:
            return QueryResult(
                instruction=inst, status="error",
                error_message=f"Cannot resolve ID: {inst.feature_id_2}",
            )

        # Always compute nominal from CAD
        if inst.query_type == QueryType.CIRCLE_DISTANCE:
            nominal = self._measure_cad_circle_distance(fid1, fid2)
        elif inst.query_type == QueryType.LINE_DISTANCE:
            nominal = self._measure_cad_line_distance(fid1, fid2)
        else:
            return QueryResult(
                instruction=inst, status="error",
                error_message="Unsupported query type",
            )

        if nominal is None:
            return QueryResult(
                instruction=inst, status="error",
                error_message="Nominal measurement computation failed",
            )

        # Try image-based measurement if registration is available
        if self._has_valid_registration and inst.query_type == QueryType.CIRCLE_DISTANCE:
            measured = self._measure_image_circle_distance(fid1, fid2)
            if measured is not None:
                return QueryResult(
                    instruction=inst,
                    value=round(measured, 4),
                    status="ok",
                    nominal=round(nominal, 4),
                    deviation=round(measured - nominal, 4),
                )

        if self._has_valid_registration and inst.query_type == QueryType.LINE_DISTANCE:
            measured = self._measure_image_line_distance(fid1, fid2)
            if measured is not None:
                return QueryResult(
                    instruction=inst,
                    value=round(measured, 4),
                    status="ok",
                    nominal=round(nominal, 4),
                    deviation=round(measured - nominal, 4),
                )

        # Fallback: CAD-only (deviation = 0)
        return QueryResult(
            instruction=inst,
            value=round(nominal, 4),
            status="ok",
            nominal=round(nominal, 4),
            deviation=0.0,
        )

    # ── Image-based measurement ──────────────────────────────────

    def _ensure_edge_cache(self) -> Optional[np.ndarray]:
        """Build the full-image Canny edge point cloud once."""
        if self._edge_cache is not None:
            return self._edge_cache
        if self._image is None or not HAS_CV2:
            return None
        from ..registration.image_extractor import ImageFeatureExtractor
        self._edge_cache = ImageFeatureExtractor.extract_edges(self._image)
        return self._edge_cache

    def _get_inv_affine(self) -> Optional[np.ndarray]:
        if self._inv_affine is not None:
            return self._inv_affine
        if self._affine is None:
            return None
        self._inv_affine = affine_solver.invert(self._affine)
        return self._inv_affine

    def _fit_circle_in_roi(
        self, cad_geometry: dict,
    ) -> Optional[Tuple[dict, float]]:
        """Fit a circle in the image ROI guided by CAD geometry.

        Returns (fitted_circle_dict, residual) or None on failure.
        """
        edges = self._ensure_edge_cache()
        inv_affine = self._get_inv_affine()
        if edges is None or inv_affine is None or len(edges) < _MIN_CIRCLE_POINTS:
            return None

        cx, cy = cad_geometry.get("cx", 0), cad_geometry.get("cy", 0)
        radius = cad_geometry.get("radius", 1.0)

        # CAD world center → pixel coords
        pixel_center = affine_solver.apply(
            inv_affine, np.array([[cx, cy]], dtype=np.float64)
        )[0]

        # Estimate pixel radius from CAD radius and affine scale
        # Use a point offset by radius along X
        offset_pt = affine_solver.apply(
            inv_affine, np.array([[cx + radius, cy]], dtype=np.float64)
        )[0]
        pixel_radius = abs(offset_pt[0] - pixel_center[0])
        if pixel_radius < 5:
            pixel_radius = 30  # reasonable fallback

        # Build ROI with padding
        roi_xmin = int(pixel_center[0] - pixel_radius - _ROI_PADDING)
        roi_ymin = int(pixel_center[1] - pixel_radius - _ROI_PADDING)
        roi_xmax = int(pixel_center[0] + pixel_radius + _ROI_PADDING)
        roi_ymax = int(pixel_center[1] + pixel_radius + _ROI_PADDING)

        from ..registration.image_extractor import ImageFeatureExtractor
        roi_edges = ImageFeatureExtractor.extract_edge_points_in_roi(
            edges, (roi_xmin, roi_ymin, roi_xmax, roi_ymax)
        )

        if len(roi_edges) < _MIN_CIRCLE_POINTS:
            return None

        fitted, residual = ImageFeatureExtractor.fit_circle_subpixel(roi_edges)
        if fitted is None or residual > _MAX_CIRCLE_RESIDUAL:
            return None

        return fitted, residual

    def _fit_line_in_roi(
        self, cad_geometry: dict,
    ) -> Optional[Tuple[dict, float]]:
        """Fit a line in the image ROI guided by CAD geometry.

        Returns (fitted_line_dict, residual) or None on failure.
        """
        edges = self._ensure_edge_cache()
        inv_affine = self._get_inv_affine()
        if edges is None or inv_affine is None or len(edges) < _MIN_LINE_POINTS:
            return None

        x1, y1 = cad_geometry.get("x1", 0), cad_geometry.get("y1", 0)
        x2, y2 = cad_geometry.get("x2", 0), cad_geometry.get("y2", 0)

        # Both endpoints → pixel coords
        pixel_pts = affine_solver.apply(
            inv_affine, np.array([[x1, y1], [x2, y2]], dtype=np.float64)
        )

        # ROI = bounding box of the two endpoints + generous padding
        px_min = min(pixel_pts[0, 0], pixel_pts[1, 0])
        px_max = max(pixel_pts[0, 0], pixel_pts[1, 0])
        py_min = min(pixel_pts[0, 1], pixel_pts[1, 1])
        py_max = max(pixel_pts[0, 1], pixel_pts[1, 1])

        # Add padding proportional to line length, with minimum
        line_len = math.sqrt((px_max - px_min) ** 2 + (py_max - py_min) ** 2)
        pad = max(_ROI_PADDING, line_len * 0.15)

        roi_xmin = int(px_min - pad)
        roi_ymin = int(py_min - pad)
        roi_xmax = int(px_max + pad)
        roi_ymax = int(py_max + pad)

        from ..registration.image_extractor import ImageFeatureExtractor
        roi_edges = ImageFeatureExtractor.extract_edge_points_in_roi(
            edges, (roi_xmin, roi_ymin, roi_xmax, roi_ymax)
        )

        if len(roi_edges) < _MIN_LINE_POINTS:
            return None

        fitted, residual = ImageFeatureExtractor.fit_line_subpixel(roi_edges)
        if fitted is None or residual > _MAX_LINE_RESIDUAL:
            return None

        return fitted, residual

    def _measure_image_circle_distance(
        self, fid1: str, fid2: str,
    ) -> Optional[float]:
        """Image-based center-to-center distance between two circles."""
        g1 = self._get_cad_geometry(fid1)
        g2 = self._get_cad_geometry(fid2)
        if g1 is None or g2 is None:
            return None

        result1 = self._fit_circle_in_roi(g1)
        result2 = self._fit_circle_in_roi(g2)
        if result1 is None or result2 is None:
            return None

        circ1, _ = result1
        circ2, _ = result2

        # Fitted pixel centers → world coords
        inv_affine = self._get_inv_affine()
        pixel_centers = np.array(
            [[circ1["cx"], circ1["cy"]], [circ2["cx"], circ2["cy"]]],
            dtype=np.float64,
        )
        # Pixel → world (forward affine)
        world_centers = affine_solver.apply(self._affine, pixel_centers)

        dx = world_centers[1, 0] - world_centers[0, 0]
        dy = world_centers[1, 1] - world_centers[0, 1]
        return math.sqrt(dx * dx + dy * dy)

    def _measure_image_line_distance(
        self, fid1: str, fid2: str,
    ) -> Optional[float]:
        """Image-based perpendicular distance between two lines."""
        g1 = self._get_cad_geometry(fid1)
        g2 = self._get_cad_geometry(fid2)
        if g1 is None or g2 is None:
            return None

        result1 = self._fit_line_in_roi(g1)
        result2 = self._fit_line_in_roi(g2)
        if result1 is None or result2 is None:
            return None

        line1, _ = result1
        line2, _ = result2

        # Fitted pixel line endpoints → world coords
        pixel_pts = np.array(
            [
                [line1["x1"], line1["y1"]],
                [line1["x2"], line1["y2"]],
                [line2["x1"], line2["y1"]],
                [line2["x2"], line2["y2"]],
            ],
            dtype=np.float64,
        )
        world_pts = affine_solver.apply(self._affine, pixel_pts)

        # Perpendicular distance: use line 1 as reference
        wx1, wy1 = world_pts[0]
        wx2, wy2 = world_pts[1]
        dx, dy = wx2 - wx1, wy2 - wy1
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-12:
            return None
        nx, ny = -dy / length, dx / length

        lx1, ly1 = world_pts[2]
        lx2, ly2 = world_pts[3]
        d1 = abs((lx1 - wx1) * nx + (ly1 - wy1) * ny)
        d2 = abs((lx2 - wx1) * nx + (ly2 - wy1) * ny)
        return (d1 + d2) / 2

    # ── CAD-only measurement functions ────────────────────────────

    def _measure_cad_circle_distance(self, fid1: str, fid2: str) -> Optional[float]:
        """Center-to-center distance between two CAD circles."""
        g1 = self._get_cad_geometry(fid1)
        g2 = self._get_cad_geometry(fid2)
        if g1 is None or g2 is None:
            return None
        cx1, cy1 = g1.get("cx", 0), g1.get("cy", 0)
        cx2, cy2 = g2.get("cx", 0), g2.get("cy", 0)
        return math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)

    def _measure_cad_line_distance(self, fid1: str, fid2: str) -> Optional[float]:
        """Perpendicular distance between two CAD lines."""
        g1 = self._get_cad_geometry(fid1)
        g2 = self._get_cad_geometry(fid2)
        if g1 is None or g2 is None:
            return None
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

    # ── ID resolution ─────────────────────────────────────────────

    def _resolve_id(self, raw_id: str) -> Optional[str]:
        """Resolve a query ID to a CADFeature.feature_id."""
        # 1. Direct feature_id match
        if self._repo.get(raw_id):
            return raw_id
        # 2. DXF handle match
        feat = self._repo.get_by_handle(raw_id)
        if feat:
            return feat.feature_id
        # 3. Partial UUID match
        for feat in self._repo.all_features():
            if feat.feature_id.startswith(raw_id):
                return feat.feature_id
        return None

    def _get_cad_geometry(self, feature_id: str) -> Optional[dict]:
        feat = self._repo.get(feature_id)
        return feat.geometry if feat else None

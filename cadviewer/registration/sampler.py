"""
CADFeatureSampler — sample CAD features into dense point clouds.

Converts CADFeature geometry dicts into Nx2 numpy arrays of world-space
points suitable for ICP registration and geometric analysis.

Sampling density is controlled by the `density` parameter (points per mm).
"""

from __future__ import annotations

import math
import numpy as np
from typing import List, Optional

from ..models.feature import CADFeature, FeatureType
from ..models.repository import FeatureRepository
from ..models.registration import RegistrationGroup


class CADFeatureSampler:
    """Sample CAD features into dense point clouds."""

    def __init__(self, default_density: float = 1.0) -> None:
        self._default_density = default_density

    def sample_feature(
        self, feature: CADFeature, density: Optional[float] = None
    ) -> np.ndarray:
        """
        Sample a single CAD feature into Nx2 points.

        Returns empty array if feature type is not samplable.
        """
        if density is None:
            density = self._default_density

        g = feature.geometry
        ft = feature.feature_type

        if ft == FeatureType.LINE:
            return self._sample_line(g, density)
        elif ft == FeatureType.CIRCLE:
            return self._sample_circle(g, density)
        elif ft == FeatureType.ARC:
            return self._sample_arc(g, density)
        elif ft == FeatureType.POLYLINE:
            return self._sample_polyline(g, density)
        elif ft == FeatureType.SPLINE:
            return self._sample_spline(g, density)
        elif ft == FeatureType.POINT:
            return np.array([[g["x"], g["y"]]], dtype=np.float64)
        else:
            return np.empty((0, 2), dtype=np.float64)

    def sample_features(
        self, features: List[CADFeature], density: Optional[float] = None
    ) -> np.ndarray:
        """Sample multiple features, concatenating into single Nx2 array."""
        points_list = []
        for feat in features:
            pts = self.sample_feature(feat, density)
            if len(pts) > 0:
                points_list.append(pts)
        if not points_list:
            return np.empty((0, 2), dtype=np.float64)
        return np.vstack(points_list)

    def sample_group(
        self,
        group: RegistrationGroup,
        repo: FeatureRepository,
        density: Optional[float] = None,
    ) -> np.ndarray:
        """Sample all features in a registration group."""
        features = [repo.get(fid) for fid in group.feature_ids]
        features = [f for f in features if f is not None]
        return self.sample_features(features, density)

    def compute_centroid(self, points: np.ndarray) -> np.ndarray:
        """Compute centroid of Nx2 point array."""
        if len(points) == 0:
            return np.zeros(2, dtype=np.float64)
        return points.mean(axis=0)

    def compute_bbox(
        self, points: np.ndarray
    ) -> tuple[float, float, float, float]:
        """Compute bounding box (min_x, min_y, max_x, max_y)."""
        if len(points) == 0:
            return (0.0, 0.0, 1.0, 1.0)
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        return (min_x, min_y, max_x, max_y)

    def compute_principal_axes(self, points: np.ndarray) -> np.ndarray:
        """Compute 2x2 rotation matrix via PCA."""
        if len(points) < 3:
            return np.eye(2, dtype=np.float64)
        centered = points - points.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        return eigenvectors[:, ::-1].T

    # ── private sampling methods ────────────────────────────────────

    def _sample_line(self, g: dict, density: float) -> np.ndarray:
        x1, y1 = g["x1"], g["y1"]
        x2, y2 = g["x2"], g["y2"]
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        n = max(2, int(length * density))
        t = np.linspace(0, 1, n)
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return np.column_stack([x, y])

    def _sample_circle(self, g: dict, density: float) -> np.ndarray:
        cx, cy, r = g["cx"], g["cy"], g["radius"]
        circumference = 2 * math.pi * r
        n = max(8, int(circumference * density))
        angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
        x = cx + r * np.cos(angles)
        y = cy + r * np.sin(angles)
        return np.column_stack([x, y])

    def _sample_arc(self, g: dict, density: float) -> np.ndarray:
        cx, cy, r = g["cx"], g["cy"], g["radius"]
        start_deg = g["start_angle"]
        end_deg = g["end_angle"]
        # Handle wrap-around
        if end_deg < start_deg:
            end_deg += 360
        arc_length = r * math.radians(end_deg - start_deg)
        n = max(4, int(arc_length * density))
        angles = np.linspace(math.radians(start_deg), math.radians(end_deg), n)
        x = cx + r * np.cos(angles)
        y = cy + r * np.sin(angles)
        return np.column_stack([x, y])

    def _sample_polyline(self, g: dict, density: float) -> np.ndarray:
        points = g.get("points", [])
        if len(points) < 2:
            return np.empty((0, 2), dtype=np.float64)
        samples = []
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            seg_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            n = max(2, int(seg_length * density))
            t = np.linspace(0, 1, n, endpoint=(i == len(points) - 2))
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            samples.append(np.column_stack([x, y]))
        if g.get("closed", False) and len(points) > 2:
            x1, y1 = points[-1]
            x2, y2 = points[0]
            seg_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            n = max(2, int(seg_length * density))
            t = np.linspace(0, 1, n, endpoint=False)
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            samples.append(np.column_stack([x, y]))
        if not samples:
            return np.empty((0, 2), dtype=np.float64)
        return np.vstack(samples)

    def _sample_spline(self, g: dict, density: float) -> np.ndarray:
        # Use fit_points if available, else control_points
        fit_pts = g.get("fit_points", [])
        ctrl_pts = g.get("control_points", [])
        pts = fit_pts if fit_pts else ctrl_pts
        if len(pts) < 2:
            return np.empty((0, 2), dtype=np.float64)
        # Simple linear interpolation along points (not true spline eval)
        return self._sample_polyline({"points": pts, "closed": False}, density)

"""
CAD silhouette extraction for global registration.

Extracts the outer contour of a CAD part for coarse alignment.
Internal features (circles, dimensions, text, hatches) are ignored.

Uses only LINE, POLYLINE, and ARC features to build an outer contour
via convex hull, then simplifies with Douglas-Peucker.
"""

from __future__ import annotations

import numpy as np

from ..models.feature import FeatureType, CADFeature

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class CADSilhouetteExtractor:
    """Extract outer contour from CAD features for registration."""

    SILHOUETTE_TYPES = {
        FeatureType.LINE,
        FeatureType.POLYLINE,
        FeatureType.ARC,
    }

    def extract_points(
        self, features: list[CADFeature], density: float = 0.5,
    ) -> np.ndarray:
        """Sample points from silhouette-relevant CAD features.

        Filters out circles, dimensions, text, hatches, splines, etc.
        Only LINE, POLYLINE, and ARC features contribute points.

        Args:
            features: list of CADFeature
            density: points per mm of curve length

        Returns:
            Nx2 float64 array in CAD world coordinates (mm)
        """
        all_pts: list[np.ndarray] = []
        for f in features:
            if f.feature_type not in self.SILHOUETTE_TYPES:
                continue
            pts = self._sample_feature(f, density)
            if pts is not None and len(pts) > 0:
                all_pts.append(pts)
        if not all_pts:
            return np.empty((0, 2), dtype=np.float64)
        return np.vstack(all_pts)

    def extract_outer_contour(
        self,
        features: list[CADFeature],
        density: float = 0.5,
        simplify_eps: float = 0.5,
    ) -> np.ndarray:
        """Extract simplified outer contour via convex hull + Douglas-Peucker.

        Args:
            features: list of CADFeature
            density: points per mm
            simplify_eps: Douglas-Peucker epsilon (mm). 0 to disable.

        Returns:
            Mx2 float64 array (ordered, closed contour)
        """
        if not HAS_CV2:
            return self.extract_points(features, density)

        points = self.extract_points(features, density)
        if len(points) < 3:
            return points

        hull = cv2.convexHull(points.astype(np.float32))
        contour = hull.reshape(-1, 2).astype(np.float64)

        if simplify_eps > 0 and len(contour) >= 3:
            cv_contour = contour.astype(np.float32).reshape(-1, 1, 2)
            simplified = cv2.approxPolyDP(cv_contour, simplify_eps, closed=True)
            contour = simplified.reshape(-1, 2).astype(np.float64)

        return contour

    # ── private sampling methods ──────────────────────────────────

    def _sample_feature(
        self, feat: CADFeature, density: float,
    ) -> np.ndarray | None:
        g = feat.geometry

        if feat.feature_type == FeatureType.LINE:
            p1 = np.array([g["x1"], g["y1"]], dtype=np.float64)
            p2 = np.array([g["x2"], g["y2"]], dtype=np.float64)
            length = np.linalg.norm(p2 - p1)
            n = max(2, int(length * density))
            t = np.linspace(0, 1, n)
            return p1 + np.outer(t, p2 - p1)

        elif feat.feature_type == FeatureType.POLYLINE:
            pts = np.array(g["points"], dtype=np.float64)
            if len(pts) < 2:
                return pts
            segs: list[np.ndarray] = []
            for i in range(len(pts) - 1):
                length = np.linalg.norm(pts[i + 1] - pts[i])
                n = max(2, int(length * density))
                t = np.linspace(0, 1, n, endpoint=(i == len(pts) - 2))
                segs.append(pts[i] + np.outer(t, pts[i + 1] - pts[i]))
            if g.get("closed", False) and len(pts) > 2:
                length = np.linalg.norm(pts[0] - pts[-1])
                n = max(2, int(length * density))
                t = np.linspace(0, 1, n, endpoint=False)
                segs.append(pts[-1] + np.outer(t, pts[0] - pts[-1]))
            return np.vstack(segs) if segs else None

        elif feat.feature_type == FeatureType.ARC:
            cx, cy, r = g["cx"], g["cy"], g["radius"]
            a0 = np.radians(g["start_angle"])
            a1 = np.radians(g["end_angle"])
            if a1 <= a0:
                a1 += 2 * np.pi
            arc_len = r * (a1 - a0)
            n = max(2, int(arc_len * density))
            angles = np.linspace(a0, a1, n)
            return np.column_stack([
                cx + r * np.cos(angles),
                cy + r * np.sin(angles),
            ])

        return None


class RegistrationContourGenerator:
    """Generate registration-ready contour from CAD features.

    Higher-level interface combining silhouette extraction with
    simplification and quality checks.
    """

    def __init__(self) -> None:
        self._extractor = CADSilhouetteExtractor()

    def generate(
        self,
        features: list[CADFeature],
        density: float = 0.5,
        simplify_eps: float = 0.5,
        min_points: int = 4,
    ) -> np.ndarray | None:
        """Generate registration contour.

        Returns contour as Mx2 array, or None if insufficient geometry.
        """
        contour = self._extractor.extract_outer_contour(
            features, density=density, simplify_eps=simplify_eps,
        )
        if len(contour) < min_points:
            return None
        return contour

    def generate_point_cloud(
        self, features: list[CADFeature], density: float = 0.5,
    ) -> np.ndarray:
        """Generate point cloud for minAreaRect computation."""
        return self._extractor.extract_points(features, density)

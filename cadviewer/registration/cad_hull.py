"""
CADHullGenerator — convex hull extraction for sparse registration.

Extracts convex hull vertices from CAD features and image contours.
Used by ConvexHullStrategy for partial-FOV registration where dense
contour ICP is unstable.
"""

from __future__ import annotations

import numpy as np

from ..models.feature import CADFeature

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from .cad_silhouette import CADSilhouetteExtractor


class CADHullGenerator:
    """Generate convex hull from CAD features for sparse registration."""

    def __init__(self) -> None:
        self._extractor = CADSilhouetteExtractor()

    def generate_hull(
        self,
        features: list[CADFeature],
        density: float = 0.5,
    ) -> np.ndarray:
        """Extract CAD convex hull vertices.

        Returns Mx2 float64 array of hull vertices (~20-50 points).
        """
        if not HAS_CV2:
            return self._extractor.extract_points(features, density)

        points = self._extractor.extract_points(features, density)
        if len(points) < 3:
            return points

        hull = cv2.convexHull(points.astype(np.float32))
        return hull.reshape(-1, 2).astype(np.float64)

    def generate_point_cloud(
        self,
        features: list[CADFeature],
        density: float = 0.5,
    ) -> np.ndarray:
        """Full point cloud for fallback / comparison."""
        return self._extractor.extract_points(features, density)


def extract_image_hull(image_contour: np.ndarray) -> np.ndarray:
    """Extract convex hull from image contour points.

    Returns Mx2 float64 hull vertices.
    """
    if not HAS_CV2 or len(image_contour) < 3:
        return image_contour

    hull = cv2.convexHull(image_contour.astype(np.float32))
    return hull.reshape(-1, 2).astype(np.float64)

"""
CorrespondenceEngine — establishes semantic CAD-to-image feature matches.

Uses predicted ROI from affine transform, feature type consistency, and
geometric constraints to match CAD features to detected image features.
Does NOT rely solely on proximity — type matching is mandatory.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Dict, List, Optional, Tuple

from ..models.feature import CADFeature, FeatureType
from ..models.repository import FeatureRepository
from ..models.image_feature import ImageFeature, ImageFeatureType, CAD_TO_IMAGE_TYPE
from ..models.correspondence import FeatureCorrespondence, CorrespondenceMap
from ..registration.affine_solver import apply, invert


class CorrespondenceEngine:
    """Computes semantic CAD-to-image feature correspondences."""

    def __init__(
        self,
        repo: FeatureRepository,
        image_repo=None,  # ImageFeatureRepository
        padding_pixels: float = 50.0,
    ) -> None:
        self._repo = repo
        self._image_repo = image_repo
        self._padding = padding_pixels

    def set_image_repo(self, image_repo) -> None:
        self._image_repo = image_repo

    def compute_correspondences(
        self,
        affine: np.ndarray,
        pixel_size_mm: float,
        cad_features: Optional[List[CADFeature]] = None,
        min_confidence: float = 0.5,
    ) -> CorrespondenceMap:
        """
        Compute correspondences between CAD and image features.

        For each CAD feature:
          1. Predict ROI in image pixel space using inverse affine
          2. Find image features of matching type inside ROI
          3. Rank by type consistency (mandatory) + geometric residual + proximity
          4. Assign best match above min_confidence
        """
        corr_map = CorrespondenceMap()

        if self._image_repo is None:
            return corr_map

        features = cad_features or self._repo.all_features()
        inv_affine = invert(affine)

        for feat in features:
            if feat.feature_type not in (
                FeatureType.LINE, FeatureType.CIRCLE, FeatureType.ARC,
                FeatureType.POLYLINE,
            ):
                continue

            # Get expected image feature type
            img_type = self._get_image_type(feat.feature_type)
            if img_type is None:
                continue

            # Predict ROI in pixel space
            roi = self._predict_roi(feat, inv_affine, pixel_size_mm)
            if roi is None:
                continue

            # Find matching image features
            candidates = self._find_candidates(img_type, roi)
            if not candidates:
                continue

            # Score and rank candidates
            best_corr = self._rank_candidates(
                feat, candidates, affine, pixel_size_mm
            )
            if best_corr and best_corr.confidence >= min_confidence:
                corr_map.add(best_corr)

        return corr_map

    def _get_image_type(self, ft: FeatureType) -> Optional[ImageFeatureType]:
        type_name = ft.name
        if type_name in CAD_TO_IMAGE_TYPE:
            return CAD_TO_IMAGE_TYPE[type_name]
        return None

    def _predict_roi(
        self,
        feat: CADFeature,
        inv_affine: np.ndarray,
        pixel_size_mm: float,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Predict ROI in image pixel space from CAD feature geometry."""
        g = feat.geometry
        ft = feat.feature_type

        # Get world-space points for the feature
        points = []
        if ft == FeatureType.LINE:
            points = [[g["x1"], g["y1"]], [g["x2"], g["y2"]]]
        elif ft in (FeatureType.CIRCLE, FeatureType.ARC):
            cx, cy, r = g["cx"], g["cy"], g.get("radius", 0)
            points = [[cx - r, cy - r], [cx + r, cy + r]]
        elif ft == FeatureType.POLYLINE:
            points = g.get("points", [])[:4]

        if not points:
            return None

        pts = np.array(points, dtype=np.float64)
        # Transform world → image pixels
        pixel_pts = apply(inv_affine, pts) / pixel_size_mm

        min_x = int(pixel_pts[:, 0].min() - self._padding)
        min_y = int(pixel_pts[:, 1].min() - self._padding)
        max_x = int(pixel_pts[:, 0].max() + self._padding)
        max_y = int(pixel_pts[:, 1].max() + self._padding)

        return (max(0, min_x), max(0, min_y), max_x, max_y)

    def _find_candidates(
        self,
        img_type: ImageFeatureType,
        roi: Tuple[int, int, int, int],
    ) -> List[ImageFeature]:
        """Find image features of matching type inside ROI."""
        if self._image_repo is None:
            return []
        candidates = []
        x_min, y_min, x_max, y_max = roi
        for img_feat in self._image_repo.features_by_type(img_type):
            g = img_feat.geometry
            # Check if feature center is inside ROI
            if img_type == ImageFeatureType.LINE:
                cx = (g.get("x1", 0) + g.get("x2", 0)) / 2
                cy = (g.get("y1", 0) + g.get("y2", 0)) / 2
            elif img_type == ImageFeatureType.CIRCLE:
                cx, cy = g.get("cx", 0), g.get("cy", 0)
            elif img_type == ImageFeatureType.CONTOUR:
                pts = g.get("points", [])
                if not pts:
                    continue
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
            else:
                continue

            if x_min <= cx <= x_max and y_min <= cy <= y_max:
                candidates.append(img_feat)
        return candidates

    def _rank_candidates(
        self,
        cad_feat: CADFeature,
        candidates: List[ImageFeature],
        affine: np.ndarray,
        pixel_size_mm: float,
    ) -> Optional[FeatureCorrespondence]:
        """Score candidates and return the best match."""
        best: Optional[FeatureCorrespondence] = None
        best_score = -1.0

        for img_feat in candidates:
            confidence, residual = self._compute_match_score(
                cad_feat, img_feat, affine, pixel_size_mm
            )
            if confidence > best_score:
                best_score = confidence
                best = FeatureCorrespondence(
                    cad_feature_id=cad_feat.feature_id,
                    image_feature_id=img_feat.feature_id,
                    confidence=confidence,
                    residual_error=residual,
                    transform_used=affine.copy(),
                    match_method="roi_type",
                )
        return best

    def _compute_match_score(
        self,
        cad_feat: CADFeature,
        img_feat: ImageFeature,
        affine: np.ndarray,
        pixel_size_mm: float,
    ) -> Tuple[float, float]:
        """
        Compute match confidence and residual.

        Score combines:
          - Type consistency (mandatory: already filtered)
          - Size similarity (radius for circles, length for lines)
          - Position proximity after alignment
        """
        g_cad = cad_feat.geometry
        g_img = img_feat.geometry
        ft = cad_feat.feature_type

        if ft == FeatureType.CIRCLE:
            cad_r = g_cad["radius"]
            img_r = g_img.get("radius", 0) * pixel_size_mm
            if cad_r < 1e-10 or img_r < 1e-10:
                return 0.0, float("inf")
            size_ratio = min(cad_r, img_r) / max(cad_r, img_r)
            # Position distance
            cad_cx, cad_cy = g_cad["cx"], g_cad["cy"]
            img_cx = g_img.get("cx", 0) * pixel_size_mm
            img_cy = g_img.get("cy", 0) * pixel_size_mm
            pos_dist = math.sqrt((cad_cx - img_cx) ** 2 + (cad_cy - img_cy) ** 2)
            residual = pos_dist + abs(cad_r - img_r)
            confidence = 0.5 * size_ratio + 0.5 * max(0, 1.0 - pos_dist / (cad_r * 2))

        elif ft == FeatureType.LINE:
            cad_len = math.sqrt(
                (g_cad["x2"] - g_cad["x1"]) ** 2 + (g_cad["y2"] - g_cad["y1"]) ** 2
            )
            img_len = math.sqrt(
                (g_img.get("x2", 0) - g_img.get("x1", 0)) ** 2 +
                (g_img.get("y2", 0) - g_img.get("y1", 0)) ** 2
            ) * pixel_size_mm
            if cad_len < 1e-10:
                return 0.0, float("inf")
            size_ratio = min(cad_len, img_len) / max(cad_len, img_len)
            residual = abs(cad_len - img_len)
            confidence = 0.6 * size_ratio + 0.4 * max(0, 1.0 - residual / cad_len)
        else:
            confidence = 0.5
            residual = 0.0

        return max(0.0, min(1.0, confidence)), residual

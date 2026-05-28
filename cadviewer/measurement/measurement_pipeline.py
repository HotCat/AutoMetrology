"""
MeasurementPipeline — orchestrates CAD-guided local feature measurement.

Flow:
  1. CAD feature → ROI prediction (via registration transform)
  2. ROI → gradient-based edge sampling (radial for circles, scanline for lines)
  3. Edge points → least-squares geometric fitting
  4. Fitted geometry → MeasuredFeature (stored separately from CAD)

CAD features are geometric priors only. The actual measurement comes
from image edge data within locally constrained search regions.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

import numpy as np

from ..models.feature import FeatureType, CADFeature
from ..models.repository import FeatureRepository
from ..models.measured_feature import MeasuredFeature, MeasuredFeatureStore
from ..registration import affine_solver
from .roi_predictor import FeatureROIPredictor, ROIRegion
from .circle_fitter import CircleFittingEngine, CircleFitResult
from .line_fitter import LineFittingEngine, LineFitResult

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def _print(msg: str) -> None:
    print(f"[MEAS] {msg}")


class MeasurementPipeline:
    """CAD-guided local feature measurement pipeline.

    Uses CAD features as geometric priors to predict ROIs, then fits
    actual geometry from image edge data.
    """

    def __init__(
        self,
        repo: FeatureRepository,
        image: np.ndarray,
        affine: np.ndarray,
        pixel_size_mm: float = 0.01,
    ) -> None:
        """
        Args:
            repo: CAD feature repository
            image: grayscale uint8 image
            affine: 3x3 matrix mapping pixel → CAD world
            pixel_size_mm: mm per pixel
        """
        self._repo = repo
        self._image = image
        self._affine = affine
        self._pixel_size_mm = pixel_size_mm

        self._store = MeasuredFeatureStore()
        self._roi_predictor = FeatureROIPredictor(affine)

        # Precompute gradient magnitude
        self._gradient: Optional[np.ndarray] = None
        if image is not None and HAS_CV2:
            grad_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
            grad_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
            self._gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)

        self._circle_engine: Optional[CircleFittingEngine] = None
        self._line_engine: Optional[LineFittingEngine] = None
        if self._gradient is not None:
            self._circle_engine = CircleFittingEngine(self._gradient)
            self._line_engine = LineFittingEngine(self._gradient)

        self._debug_data: dict = {}

    @property
    def store(self) -> MeasuredFeatureStore:
        return self._store

    def measure_feature(self, cad_feature_id: str) -> Optional[MeasuredFeature]:
        """Measure a single feature. Returns cached result if available."""
        # Check cache
        existing = self._store.get_by_cad_id(cad_feature_id)
        if existing is not None:
            return existing

        feat = self._repo.get(cad_feature_id)
        if feat is None:
            return None

        if feat.feature_type == FeatureType.CIRCLE:
            return self._measure_circle(feat)
        elif feat.feature_type == FeatureType.LINE:
            return self._measure_line(feat)

        return None

    def measure_features(
        self, cad_feature_ids: list[str],
    ) -> list[MeasuredFeature]:
        """Measure multiple features."""
        results = []
        for fid in cad_feature_ids:
            mf = self.measure_feature(fid)
            if mf is not None:
                results.append(mf)
        return results

    def measure_all(self) -> list[MeasuredFeature]:
        """Measure all measurable features (CIRCLE and LINE) in the repo."""
        _print("Measuring all features...")
        results = []
        for feat in self._repo.all_features():
            if feat.feature_type == FeatureType.CIRCLE:
                mf = self.measure_feature(feat.feature_id)
                if mf is not None:
                    results.append(mf)
            elif feat.feature_type == FeatureType.LINE:
                mf = self.measure_feature(feat.feature_id)
                if mf is not None:
                    results.append(mf)
        _print(f"  Measured {len(results)} features")
        return results

    def get_debug_data(self) -> dict:
        return self._debug_data

    # ── private measurement methods ──────────────────────────────

    def _measure_circle(self, feat: CADFeature) -> Optional[MeasuredFeature]:
        """Measure a circle feature via radial edge sampling."""
        if self._circle_engine is None:
            return None

        geom = feat.geometry
        roi_result = self._roi_predictor.predict_circle_roi(geom, padding=15)
        if roi_result is None:
            return None
        roi, pixel_center, pixel_radius = roi_result

        # Fit
        result: Optional[CircleFitResult] = self._circle_engine.fit(
            pixel_center, pixel_radius,
        )
        if result is None:
            return None

        # Convert fitted center and radius to world coords
        pixel_center_fitted = np.array([[result.center[0], result.center[1]]])
        world_center = affine_solver.apply(self._affine, pixel_center_fitted)[0]

        # Convert radius to world coords
        pixel_edge = np.array([[result.center[0] + result.radius, result.center[1]]])
        world_edge = affine_solver.apply(self._affine, pixel_edge)[0]
        world_radius = float(np.linalg.norm(world_edge - world_center))

        fitted_geom = {
            "cx": float(result.center[0]),
            "cy": float(result.center[1]),
            "radius": float(result.radius),
        }
        fitted_geom_world = {
            "cx": float(world_center[0]),
            "cy": float(world_center[1]),
            "radius": world_radius,
        }

        mf = MeasuredFeature(
            feature_id=str(uuid.uuid4()),
            cad_feature_id=feat.feature_id,
            feature_type=FeatureType.CIRCLE,
            fitted_geometry=fitted_geom,
            fitted_geometry_world=fitted_geom_world,
            edge_points=result.edge_points,
            roi_bbox=(roi.xmin, roi.ymin, roi.xmax, roi.ymax),
            residual_error=result.residual,
            confidence=result.confidence,
            detection_method="radial_edge_sampling",
        )
        self._store.add(mf)

        # Store debug data
        self._debug_data[feat.feature_id] = {
            "type": "circle",
            "roi": (roi.xmin, roi.ymin, roi.xmax, roi.ymax),
            "predicted_center": pixel_center,
            "predicted_radius": pixel_radius,
            "edge_points": result.edge_points,
            "fitted_center": result.center,
            "fitted_radius": result.radius,
            "residual": result.residual,
            "confidence": result.confidence,
        }

        return mf

    def _measure_line(self, feat: CADFeature) -> Optional[MeasuredFeature]:
        """Measure a line feature via perpendicular scanline sampling."""
        if self._line_engine is None:
            return None

        geom = feat.geometry
        roi_result = self._roi_predictor.predict_line_roi(geom, padding=15)
        if roi_result is None:
            return None
        roi, pixel_p1, pixel_p2 = roi_result

        # Fit
        result: Optional[LineFitResult] = self._line_engine.fit(
            pixel_p1, pixel_p2,
        )
        if result is None:
            return None

        # Convert fitted line endpoints to world coords
        pixel_pts = np.array([result.p1, result.p2])
        world_pts = affine_solver.apply(self._affine, pixel_pts)

        fitted_geom = {
            "x1": float(result.p1[0]),
            "y1": float(result.p1[1]),
            "x2": float(result.p2[0]),
            "y2": float(result.p2[1]),
        }
        fitted_geom_world = {
            "x1": float(world_pts[0, 0]),
            "y1": float(world_pts[0, 1]),
            "x2": float(world_pts[1, 0]),
            "y2": float(world_pts[1, 1]),
        }

        mf = MeasuredFeature(
            feature_id=str(uuid.uuid4()),
            cad_feature_id=feat.feature_id,
            feature_type=FeatureType.LINE,
            fitted_geometry=fitted_geom,
            fitted_geometry_world=fitted_geom_world,
            edge_points=result.edge_points,
            roi_bbox=(roi.xmin, roi.ymin, roi.xmax, roi.ymax),
            residual_error=result.residual,
            confidence=result.confidence,
            detection_method="perpendicular_scanline",
        )
        self._store.add(mf)

        self._debug_data[feat.feature_id] = {
            "type": "line",
            "roi": (roi.xmin, roi.ymin, roi.xmax, roi.ymax),
            "predicted_p1": pixel_p1,
            "predicted_p2": pixel_p2,
            "edge_points": result.edge_points,
            "fitted_p1": result.p1,
            "fitted_p2": result.p2,
            "residual": result.residual,
            "confidence": result.confidence,
        }

        return mf

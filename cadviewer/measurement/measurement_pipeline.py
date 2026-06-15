"""
MeasurementPipeline — orchestrates CAD-guided local feature measurement.

DATA CONTRACT:
  1. Image source is ALWAYS raw camera image, never CAD overlay or display canvas
  2. MeasuredFeature.source_type is set to "FITTED" (image-derived geometry)
  3. Query evaluator uses MeasuredFeature.fitted_geometry_world ONLY

Flow:
  1. CAD feature → ROI prediction (via registration transform)
  2. ROI → gradient-based edge sampling (radial for circles, scanline for lines)
  3. Edge points → least-squares geometric fitting
  4. Fitted geometry → MeasuredFeature (source_type="FITTED")

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
from ..calibration.residual_map import ResidualDistortionMap
from .roi_predictor import FeatureROIPredictor, ROIRegion
from .circle_fitter import CircleFittingEngine, CircleFitResult
from .line_fitter import LineFittingEngine, LineFitResult

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import diplib as dip
    HAS_DIP = True
except ImportError:
    HAS_DIP = False


def _print(msg: str) -> None:
    print(f"[MEAS] {msg}")


def _audit(msg: str) -> None:
    logger.info(f"[AUDIT] {msg}")


class MeasurementPipeline:
    """CAD-guided local feature measurement pipeline.

    Uses CAD features as geometric priors to predict ROIs, then fits
    actual geometry from image edge data.

    IMAGE SOURCE CONTRACT:
      The gradient image passed to fitting engines is computed from
      self._image, which is the raw grayscale camera image.
      There is no code path that uses CAD overlay pixels or display
      canvas pixels as input to measurement.
    """

    def __init__(
        self,
        repo: FeatureRepository,
        image: np.ndarray,
        affine: np.ndarray,
        pixel_size_mm: float = 0.01,
        residual_map: Optional[ResidualDistortionMap] = None,
        pixel_to_world_transform: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            repo: CAD feature repository
            image: grayscale uint8 image (RAW CAMERA IMAGE, not overlay)
            affine: 3x3 matrix mapping pixel → CAD world
            pixel_size_mm: mm per pixel
            residual_map: optional residual distortion map for sub-pixel correction
            pixel_to_world_transform: optional 3x3 affine/projective matrix used
                for ROI prediction and final fitted pixel -> CAD world geometry
                after safety validation.
        """
        self._repo = repo
        self._image = image
        self._affine = affine
        self._pixel_size_mm = pixel_size_mm
        self._residual_map = residual_map
        self._pixel_to_world_transform = None
        if pixel_to_world_transform is not None:
            candidate = np.asarray(pixel_to_world_transform, dtype=np.float64)
            image_size = None
            if image is not None:
                image_size = (int(image.shape[1]), int(image.shape[0]))
            try:
                from ..calibration.transform_safety import validate_pixel_to_world_transform
                safety = validate_pixel_to_world_transform(
                    candidate, float(pixel_size_mm), image_size=image_size,
                )
                if safety.safe:
                    self._pixel_to_world_transform = candidate
                else:
                    _audit(f"Rejected measurement transform: {safety.reason}")
            except Exception as exc:
                _audit(f"Rejected measurement transform: {exc}")

        # Image source assertion: must be raw camera image
        self._assert_image_source(image)

        self._store = MeasuredFeatureStore()
        roi_transform = (
            self._pixel_to_world_transform
            if self._pixel_to_world_transform is not None else self._affine
        )
        self._roi_predictor = FeatureROIPredictor(roi_transform)

        # Precompute gradient magnitude from RAW IMAGE
        self._gradient: Optional[np.ndarray] = None
        if image is not None and HAS_CV2:
            grad_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
            grad_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
            self._gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)
            _audit(f"Gradient computed from image: shape={image.shape}, dtype={image.dtype}")

        self._circle_engine: Optional[CircleFittingEngine] = None
        self._line_engine: Optional[LineFittingEngine] = None
        if self._gradient is not None:
            self._circle_engine = CircleFittingEngine(self._gradient)
            self._line_engine = LineFittingEngine(self._gradient)
            _audit(f"Fitting engines initialized with gradient from RAW IMAGE")

        self._debug_data: dict = {}

    def _assert_image_source(self, image: np.ndarray) -> None:
        """Assert that image is raw camera data, not CAD overlay.

        This is a structural assertion — the image passed to MeasurementPipeline
        comes from ImageLayerRenderer.image (numpy BGR array from camera or file),
        not from any CAD rendering or display compositing.
        """
        if image is None:
            return
        # Type check: must be uint8 grayscale (converted from BGR camera image)
        assert image.dtype == np.uint8, (
            f"Image source assertion failed: dtype={image.dtype}, expected uint8. "
            f"Image must be raw camera data converted to grayscale."
        )
        assert len(image.shape) == 2, (
            f"Image source assertion failed: shape={image.shape}, expected 2D grayscale. "
            f"Image must be raw camera data converted to grayscale, not BGR or RGBA."
        )
        _audit(f"Image source validated: dtype=uint8, shape={image.shape} (RAW CAMERA)")

    @property
    def store(self) -> MeasuredFeatureStore:
        return self._store

    @property
    def measurement_transform(self) -> np.ndarray:
        """Return the transform used for measurement/world conversion."""
        return (
            self._pixel_to_world_transform
            if self._pixel_to_world_transform is not None else self._affine
        )

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
        elif feat.feature_type == FeatureType.ARC:
            return self._measure_arc(feat)

        return None

    def measure_line_pair(
        self, cad_feature_id_1: str, cad_feature_id_2: str,
    ) -> tuple[Optional[MeasuredFeature], Optional[MeasuredFeature]]:
        """Measure a line pair with deterministic inward edge preference.

        Adjacent printed/etched lines often produce two parallel gradient peaks.
        For pair distance queries, the useful edge is the one facing the paired
        CAD line, which also minimizes the pair distance without depending on
        small registration drift. Pair fits are not cached globally because the
        preferred side is query-context specific.
        """
        feat1 = self._repo.get(cad_feature_id_1)
        feat2 = self._repo.get(cad_feature_id_2)
        if feat1 is None or feat2 is None:
            return None, None
        if feat1.feature_type != FeatureType.LINE or feat2.feature_type != FeatureType.LINE:
            return self.measure_feature(cad_feature_id_1), self.measure_feature(cad_feature_id_2)
        return (
            self._measure_line(feat1, paired_geometry=feat2.geometry, cache=False),
            self._measure_line(feat2, paired_geometry=feat1.geometry, cache=False),
        )

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
            elif feat.feature_type == FeatureType.ARC:
                mf = self.measure_feature(feat.feature_id)
                if mf is not None:
                    results.append(mf)
        _print(f"  Measured {len(results)} features")
        return results

    def get_debug_data(self) -> dict:
        return self._debug_data

    def _correct_points(self, points: np.ndarray) -> np.ndarray:
        """Apply residual distortion correction if map is available."""
        if self._residual_map is not None and self._residual_map.is_built:
            return self._residual_map.correct(points)
        return points

    def _pixel_points_to_world(self, points: np.ndarray) -> np.ndarray:
        """Transform fitted pixel points to CAD world coordinates."""
        corrected = self._correct_points(points)
        if self._pixel_to_world_transform is None:
            return affine_solver.apply_projective(self._affine, corrected)

        return affine_solver.apply_projective(self._pixel_to_world_transform, corrected)

    # ── private measurement methods ──────────────────────────────

    def _measure_circle(self, feat: CADFeature) -> Optional[MeasuredFeature]:
        """Measure a circle feature via radial edge sampling."""
        if self._circle_engine is None:
            return None

        geom = feat.geometry
        roi_result = self._roi_predictor.predict_circle_roi(geom, padding=50)
        if roi_result is None:
            return None
        roi, pixel_center, pixel_radius = roi_result

        # Primary radial fit, then a constrained Canny/Hough fallback for
        # small filled/hollow circles where radial sampling can hit the wrong
        # edge or return too few points.
        result: Optional[CircleFitResult] = self._circle_engine.fit(
            pixel_center, pixel_radius,
            n_rays=180,
            search_width_ratio=0.75,
            min_gradient=12.0,
        )
        if result is None or self._circle_fit_needs_fallback(result, pixel_radius):
            fallback = self._measure_circle_with_edge_detector(
                pixel_center, pixel_radius,
            )
            if fallback is not None and (
                result is None or fallback.confidence >= result.confidence
            ):
                result = fallback

        local_result = self._measure_circle_with_local_detector(
            pixel_center, pixel_radius,
        )
        if local_result is not None and (
            result is None
            or self._prefer_local_circle_fit(local_result, result, pixel_radius)
        ):
            result = local_result

        if result is None:
            _print(f"  Circle {feat.feature_id[:12]}: NO EDGE FOUND "
                   f"(predicted=({pixel_center[0]:.1f},{pixel_center[1]:.1f}) "
                   f"r={pixel_radius:.1f}px)")
            return None

        # Log pixel-space displacement from predicted to fitted
        dp = result.center - pixel_center
        displacement_px = float(np.linalg.norm(dp))
        _print(f"  Circle {feat.feature_id[:12]}: "
               f"predicted=({pixel_center[0]:.1f},{pixel_center[1]:.1f}) "
               f"fitted=({result.center[0]:.1f},{result.center[1]:.1f}) "
               f"Δpx=({dp[0]:.1f},{dp[1]:.1f}) "
               f"conf={result.confidence:.2f} pts={result.n_edge_points}")

        # Gradient quality validation: require strong edges
        image_grad_mean = float(np.mean(self._gradient)) if self._gradient is not None else 0.0
        if result.gradient_strength < max(45.0, image_grad_mean * 3.0):
            _print(f"  REJECTED: gradient_strength={result.gradient_strength:.1f} < threshold "
                   f"(edges too weak, likely noise)")
            return None

        # Convert fitted center and radius to world coords
        # Apply residual distortion correction first
        pixel_center_fitted = np.array([[result.center[0], result.center[1]]])
        world_center = self._pixel_points_to_world(pixel_center_fitted)[0]

        # Convert radius to world coords
        pixel_edge = np.array([[result.center[0] + result.radius, result.center[1]]])
        world_edge = self._pixel_points_to_world(pixel_edge)[0]
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

        roi_bbox = getattr(
            result, "roi_bbox", (roi.xmin, roi.ymin, roi.xmax, roi.ymax),
        )

        mf = MeasuredFeature(
            feature_id=str(uuid.uuid4()),
            cad_feature_id=feat.feature_id,
            feature_type=FeatureType.CIRCLE,
            fitted_geometry=fitted_geom,
            fitted_geometry_world=fitted_geom_world,
            edge_points=result.edge_points,
            roi_bbox=roi_bbox,
            residual_error=result.residual,
            confidence=result.confidence,
            detection_method=getattr(result, "method", "radial_edge_sampling"),
            source_type="FITTED",
        )
        self._store.add(mf)

        # Store debug data
        self._debug_data[feat.feature_id] = {
            "type": "circle",
            "roi": roi_bbox,
            "predicted_center": pixel_center,
            "predicted_radius": pixel_radius,
            "edge_points": result.edge_points,
            "fitted_center": result.center,
            "fitted_radius": result.radius,
            "residual": result.residual,
            "confidence": result.confidence,
            "detection_method": getattr(result, "method", "radial_edge_sampling"),
        }

        return mf

    @staticmethod
    def _prefer_local_circle_fit(
        local_result: CircleFitResult,
        current_result: CircleFitResult,
        predicted_radius: float,
    ) -> bool:
        method = getattr(local_result, "method", "")
        if not str(method).startswith("local_dip_watershed"):
            return False
        if local_result.confidence < 0.65:
            return False
        if local_result.radius < predicted_radius * 0.45:
            return False
        if local_result.radius > predicted_radius * 1.80:
            return False
        if predicted_radius <= 25.0:
            local_radius_error = (
                abs(local_result.radius - predicted_radius)
                / max(predicted_radius, 1.0)
            )
            current_radius_error = (
                abs(current_result.radius - predicted_radius)
                / max(predicted_radius, 1.0)
            )
            if (
                local_radius_error + 0.08 < current_radius_error
                and local_result.confidence >= current_result.confidence - 0.20
            ):
                return True
            return (
                local_result.confidence >= current_result.confidence - 0.15
                and local_radius_error <= current_radius_error + 0.10
            )
        return (
            current_result.residual > max(2.0, predicted_radius * 0.14)
            and local_result.confidence >= current_result.confidence - 0.05
        )

    @staticmethod
    def _circle_fit_needs_fallback(
        result: CircleFitResult, predicted_radius: float,
    ) -> bool:
        if result.confidence < 0.35:
            return True
        if result.residual > max(2.0, predicted_radius * 0.18):
            return True
        if result.radius < predicted_radius * 0.45:
            return True
        if result.radius > predicted_radius * 2.2:
            return True
        return False

    def _measure_circle_with_edge_detector(
        self, pixel_center: np.ndarray, pixel_radius: float,
    ) -> Optional[CircleFitResult]:
        """Fit small filled/hollow circles from local Canny edges."""
        if not HAS_CV2 or self._image is None or self._gradient is None:
            return None

        h, w = self._image.shape[:2]
        padding = max(28.0, pixel_radius * 1.8)
        xmin = max(0, int(pixel_center[0] - pixel_radius - padding))
        ymin = max(0, int(pixel_center[1] - pixel_radius - padding))
        xmax = min(w, int(pixel_center[0] + pixel_radius + padding))
        ymax = min(h, int(pixel_center[1] + pixel_radius + padding))
        if xmax - xmin < 8 or ymax - ymin < 8:
            return None

        crop = self._image[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            return None
        blurred = cv2.GaussianBlur(crop, (3, 3), 0)
        edges = cv2.Canny(blurred, 20, 80, L2gradient=True)
        ys, xs = np.nonzero(edges)
        if len(xs) < 8:
            return None

        edge_points = np.column_stack([xs + xmin, ys + ymin]).astype(np.float64)
        best = self._best_hough_circle_from_edges(
            blurred, edge_points, pixel_center, pixel_radius, xmin, ymin,
        )
        if best is None:
            best = self._best_annulus_circle_from_edges(
                edge_points, pixel_center, pixel_radius,
            )
        if best is None:
            return None

        center, radius, support_points = best
        if len(support_points) < 8:
            return None

        dists = np.sqrt(
            (support_points[:, 0] - center[0]) ** 2
            + (support_points[:, 1] - center[1]) ** 2
        )
        residual = float(np.mean(np.abs(dists - radius)))
        if residual > max(3.0, radius * 0.30):
            return None

        displacement = float(np.linalg.norm(center - pixel_center))
        max_displacement = max(38.0, pixel_radius * 3.0)
        if displacement > max_displacement:
            return None
        if radius < max(2.0, pixel_radius * 0.30) or radius > pixel_radius * 2.2:
            return None

        ix = np.clip(np.round(support_points[:, 0]).astype(int), 0, w - 1)
        iy = np.clip(np.round(support_points[:, 1]).astype(int), 0, h - 1)
        gradient_strength = float(np.mean(self._gradient[iy, ix]))
        image_grad_mean = float(np.mean(self._gradient))
        if gradient_strength < max(25.0, image_grad_mean * 1.6):
            return None

        residual_score = max(0.0, 1.0 - residual / max(2.5, radius * 0.25))
        displacement_score = max(0.0, 1.0 - displacement / max_displacement)
        support_score = min(1.0, len(support_points) / 36.0)
        confidence = float(
            max(0.0, min(1.0, 0.55 * residual_score
                         + 0.25 * support_score
                         + 0.20 * displacement_score))
        )

        _print(
            "  Circle edge fit: "
            f"predicted=({pixel_center[0]:.1f},{pixel_center[1]:.1f}) "
            f"fitted=({center[0]:.1f},{center[1]:.1f}) "
            f"r={radius:.1f}px Δ={displacement:.1f}px "
            f"conf={confidence:.2f} pts={len(support_points)}"
        )

        result = CircleFitResult(
            center=center,
            radius=float(radius),
            edge_points=support_points,
            residual=residual,
            confidence=confidence,
            n_edge_points=len(support_points),
            gradient_strength=gradient_strength,
        )
        result.roi_bbox = (xmin, ymin, xmax, ymax)
        result.method = "canny_hough_circle"
        return result

    def _best_hough_circle_from_edges(
        self,
        crop: np.ndarray,
        edge_points: np.ndarray,
        pixel_center: np.ndarray,
        pixel_radius: float,
        xmin: int,
        ymin: int,
    ) -> Optional[tuple[np.ndarray, float, np.ndarray]]:
        min_radius = max(3, int(pixel_radius * 0.35))
        max_radius = max(min_radius + 2, int(pixel_radius * 2.1))
        circles = cv2.HoughCircles(
            crop,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=max(8, int(pixel_radius)),
            param1=60,
            param2=7,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is None:
            return None

        candidates = []
        for cx, cy, radius in circles[0]:
            center = np.array([float(cx + xmin), float(cy + ymin)], dtype=np.float64)
            radius = float(radius)
            support = self._circle_support_points(edge_points, center, radius)
            if len(support) < 8:
                continue
            center_offset = float(np.linalg.norm(center - pixel_center))
            radius_error = abs(radius - pixel_radius) / max(pixel_radius, 1.0)
            support_score = min(1.0, len(support) / 40.0)
            score = support_score - 0.020 * center_offset - 0.25 * radius_error
            candidates.append((score, center, radius, support))

        if not candidates:
            return None
        _, center, radius, support = max(candidates, key=lambda item: item[0])
        refined = self._refine_circle_from_support(support, center, radius)
        return refined if refined is not None else (center, radius, support)

    def _best_annulus_circle_from_edges(
        self,
        edge_points: np.ndarray,
        pixel_center: np.ndarray,
        pixel_radius: float,
    ) -> Optional[tuple[np.ndarray, float, np.ndarray]]:
        dists = np.sqrt(
            (edge_points[:, 0] - pixel_center[0]) ** 2
            + (edge_points[:, 1] - pixel_center[1]) ** 2
        )
        band = max(7.0, pixel_radius * 0.80)
        points = edge_points[
            (dists >= pixel_radius - band)
            & (dists <= pixel_radius + band)
        ]
        if len(points) < 8:
            return None
        fitted = CircleFittingEngine._fit_circle_kasa(points)
        if fitted is None:
            return None
        center = np.array([fitted[0], fitted[1]], dtype=np.float64)
        radius = float(fitted[2])
        support = self._circle_support_points(edge_points, center, radius)
        if len(support) < 8:
            support = points
        return self._refine_circle_from_support(support, center, radius)

    @staticmethod
    def _circle_support_points(
        edge_points: np.ndarray, center: np.ndarray, radius: float,
    ) -> np.ndarray:
        dists = np.sqrt(
            (edge_points[:, 0] - center[0]) ** 2
            + (edge_points[:, 1] - center[1]) ** 2
        )
        band = max(2.5, radius * 0.28)
        return edge_points[np.abs(dists - radius) <= band]

    @staticmethod
    def _refine_circle_from_support(
        support_points: np.ndarray,
        center: np.ndarray,
        radius: float,
    ) -> Optional[tuple[np.ndarray, float, np.ndarray]]:
        if len(support_points) < 8:
            return None
        fitted = CircleFittingEngine._fit_circle_kasa(support_points)
        if fitted is None:
            return center, radius, support_points
        refined_center = np.array([fitted[0], fitted[1]], dtype=np.float64)
        refined_radius = float(fitted[2])
        # Kasa is unstable for short/noisy arcs; keep the Hough center if the
        # refinement jumps too far.
        if np.linalg.norm(refined_center - center) > max(6.0, radius * 0.60):
            return center, radius, support_points
        if refined_radius < radius * 0.45 or refined_radius > radius * 1.8:
            return center, radius, support_points
        refined_support = MeasurementPipeline._circle_support_points(
            support_points, refined_center, refined_radius,
        )
        if len(refined_support) >= 8:
            support_points = refined_support
        return refined_center, refined_radius, support_points

    def _measure_circle_with_local_detector(
        self, pixel_center: np.ndarray, pixel_radius: float,
    ) -> Optional[CircleFitResult]:
        """Fallback for small/filled fiducials where radial sampling is brittle."""
        if not HAS_CV2 or self._image is None or self._gradient is None:
            return None

        from ..registration.auto_correspondence import detect_circle_in_roi

        h, w = self._image.shape[:2]
        padding = max(35.0, pixel_radius * 2.5)
        xmin = max(0, int(pixel_center[0] - pixel_radius - padding))
        ymin = max(0, int(pixel_center[1] - pixel_radius - padding))
        xmax = min(w, int(pixel_center[0] + pixel_radius + padding))
        ymax = min(h, int(pixel_center[1] + pixel_radius + padding))
        if xmax - xmin < 8 or ymax - ymin < 8:
            return None

        detection = detect_circle_in_roi(
            self._image,
            (xmin, ymin, xmax - xmin, ymax - ymin),
            expected_radius_px=pixel_radius,
            expected_center=(float(pixel_center[0]), float(pixel_center[1])),
        )
        if detection is None:
            return None

        center = np.array(detection.center, dtype=np.float64)
        radius = float(detection.radius)
        displacement = float(np.linalg.norm(center - pixel_center))
        max_displacement = max(42.0, padding + pixel_radius)
        if displacement > max_displacement:
            return None
        if radius < max(2.0, pixel_radius * 0.35) or radius > pixel_radius * 2.5:
            return None

        edge_points = self._circle_edge_points_from_detection(
            center, radius, xmin, ymin, xmax, ymax,
        )
        if len(edge_points) == 0:
            angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
            edge_points = np.column_stack([
                center[0] + radius * np.cos(angles),
                center[1] + radius * np.sin(angles),
            ])

        dists = np.sqrt(
            (edge_points[:, 0] - center[0]) ** 2
            + (edge_points[:, 1] - center[1]) ** 2
        )
        residual = float(np.mean(np.abs(dists - radius)))
        residual_score = max(0.25, 1.0 - residual / max(radius * 0.35, 2.0))
        if str(detection.method).startswith("dip_watershed"):
            # DIPLib MeasurementTool already measured roundness/radius/gravity
            # from the labeled fiducial. Canny points here are only an overlay
            # audit and can be sparse on filled fiducials, so do not let them
            # erase an otherwise strong watershed measurement.
            residual_score = max(0.75, residual_score)
        confidence = float(min(1.0, detection.confidence * residual_score))

        ix = np.clip(np.round(edge_points[:, 0]).astype(int), 0, w - 1)
        iy = np.clip(np.round(edge_points[:, 1]).astype(int), 0, h - 1)
        gradient_strength = float(np.mean(self._gradient[iy, ix]))

        _print(
            "  Circle fallback: "
            f"predicted=({pixel_center[0]:.1f},{pixel_center[1]:.1f}) "
            f"detected=({center[0]:.1f},{center[1]:.1f}) "
            f"r={radius:.1f}px Δ={displacement:.1f}px "
            f"method={detection.method} conf={confidence:.2f}"
        )

        result = CircleFitResult(
            center=center,
            radius=radius,
            edge_points=edge_points,
            residual=residual,
            confidence=confidence,
            n_edge_points=len(edge_points),
            gradient_strength=gradient_strength,
        )
        result.roi_bbox = (xmin, ymin, xmax, ymax)
        result.method = f"local_{detection.method}"
        return result

    def _circle_edge_points_from_detection(
        self,
        center: np.ndarray,
        radius: float,
        xmin: int,
        ymin: int,
        xmax: int,
        ymax: int,
    ) -> np.ndarray:
        """Collect image edge pixels near a detected circle for audit/overlay."""
        crop = self._image[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            return np.empty((0, 2), dtype=np.float64)

        if crop.ndim == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        edges = cv2.Canny(gray, 40, 140)
        ys, xs = np.nonzero(edges)
        if len(xs) == 0:
            return np.empty((0, 2), dtype=np.float64)

        pts = np.column_stack([xs + xmin, ys + ymin]).astype(np.float64)
        dists = np.sqrt(
            (pts[:, 0] - center[0]) ** 2 + (pts[:, 1] - center[1]) ** 2
        )
        band = max(2.5, radius * 0.30)
        pts = pts[np.abs(dists - radius) <= band]
        return pts

    def _measure_arc(self, feat: CADFeature) -> Optional[MeasuredFeature]:
        """Measure an arc radius by fitting image edges near the CAD arc."""
        if not HAS_CV2 or self._image is None or self._gradient is None:
            return None

        geom = feat.geometry
        arc_world = self._sample_arc_world_points(geom)
        if len(arc_world) < 4:
            return None

        world_to_pixel = affine_solver.invert(
            self._pixel_to_world_transform
            if self._pixel_to_world_transform is not None else self._affine
        )
        arc_px = affine_solver.apply_projective(world_to_pixel, arc_world)
        center_world = np.array([[geom["cx"], geom["cy"]]], dtype=np.float64)
        pixel_center = affine_solver.apply_projective(world_to_pixel, center_world)[0]
        pixel_radius = float(np.mean(np.linalg.norm(arc_px - pixel_center, axis=1)))
        if pixel_radius < 3.0:
            return None

        h, w = self._image.shape[:2]
        padding = max(80.0, pixel_radius * 0.90)
        xmin = max(0, int(np.floor(np.min(arc_px[:, 0]) - padding)))
        ymin = max(0, int(np.floor(np.min(arc_px[:, 1]) - padding)))
        xmax = min(w, int(np.ceil(np.max(arc_px[:, 0]) + padding)))
        ymax = min(h, int(np.ceil(np.max(arc_px[:, 1]) + padding)))
        if xmax - xmin < 8 or ymax - ymin < 8:
            return None

        crop = self._image[ymin:ymax, xmin:xmax]
        edges = cv2.Canny(crop, 40, 140)
        ys, xs = np.nonzero(edges)
        if len(xs) < 8:
            _print(f"  Arc {feat.feature_id[:12]}: NO EDGE FOUND in ROI")
            return None

        edge_points = np.column_stack([xs + xmin, ys + ymin]).astype(np.float64)
        search_width = max(6.0, pixel_radius * 0.20)
        edge_points = self._filter_points_near_polyline(
            edge_points, arc_px, search_width, endpoint_margin_ratio=0.08,
        )
        if len(edge_points) < 8:
            _print(f"  Arc {feat.feature_id[:12]}: insufficient edge support "
                   f"near projected CAD arc ({len(edge_points)} pts)")
            return None

        center = pixel_center.astype(np.float64)
        # Prefer radial profiles for arcs so the inner edge along the CAD
        # radius is considered before larger, high-support outer contours.
        arc_fit = self._fit_arc_radius_from_radial_edges(
            center, pixel_radius, arc_px,
        )
        if arc_fit is None:
            arc_fit = self._fit_arc_radius_from_dip_contours(
                center, pixel_radius, arc_px, (xmin, ymin, xmax, ymax),
            )
        if arc_fit is not None:
            radius = float(arc_fit["radius"])
            center = np.asarray(arc_fit.get("center", center), dtype=np.float64)
            edge_points = arc_fit["edge_points"]
            residual = float(arc_fit["residual"])
            gradient_strength = float(arc_fit["gradient_strength"])
            arc_fit_method = str(arc_fit.get("method", "arc_radial_profile"))
        else:
            # Fallback for very low-support arcs: keep the CAD-guided center and
            # derive radius from image edge points around that center. This is
            # less selective than radial-profile fitting, so it is used only
            # when the profile sampler cannot find enough edge peaks.
            dists = np.sqrt(
                (edge_points[:, 0] - center[0]) ** 2
                + (edge_points[:, 1] - center[1]) ** 2
            )
            radius = float(np.median(dists))
            residual = float(np.mean(np.abs(dists - radius)))
            ix = np.clip(np.round(edge_points[:, 0]).astype(int), 0, w - 1)
            iy = np.clip(np.round(edge_points[:, 1]).astype(int), 0, h - 1)
            gradient_strength = float(np.mean(self._gradient[iy, ix]))
            arc_fit_method = "arc_edge_sampling"

        if residual > max(5.0, pixel_radius * 0.15):
            _print(f"  Arc {feat.feature_id[:12]}: rejected residual={residual:.2f}px")
            return None

        image_grad_mean = float(np.mean(self._gradient))
        if gradient_strength < max(35.0, image_grad_mean * 2.0):
            _print(f"  Arc {feat.feature_id[:12]}: rejected weak edges "
                   f"gradient={gradient_strength:.1f}")
            return None

        residual_score = max(0.0, 1.0 - residual / max(pixel_radius * 0.12, 2.0))
        support_target = 8.0 if arc_fit is not None else max(20.0, len(arc_px) * 0.35)
        support_score = min(1.0, len(edge_points) / support_target)
        prior_score = max(0.0, 1.0 - abs(radius - pixel_radius) / max(pixel_radius * 0.30, 8.0))
        confidence = float(max(0.0, min(
            1.0,
            0.45 * residual_score + 0.35 * support_score + 0.20 * prior_score,
        )))

        world_center = self._pixel_points_to_world(center.reshape(1, 2))[0]
        pixel_edge = np.array([[center[0] + radius, center[1]]], dtype=np.float64)
        world_edge = self._pixel_points_to_world(pixel_edge)[0]
        world_radius = float(np.linalg.norm(world_edge - world_center))

        fitted_geom = {
            "cx": float(center[0]),
            "cy": float(center[1]),
            "radius": float(radius),
            "start_angle": geom.get("start_angle"),
            "end_angle": geom.get("end_angle"),
        }
        fitted_geom_world = {
            "cx": float(world_center[0]),
            "cy": float(world_center[1]),
            "radius": world_radius,
            "start_angle": geom.get("start_angle"),
            "end_angle": geom.get("end_angle"),
        }

        mf = MeasuredFeature(
            feature_id=str(uuid.uuid4()),
            cad_feature_id=feat.feature_id,
            feature_type=FeatureType.ARC,
            fitted_geometry=fitted_geom,
            fitted_geometry_world=fitted_geom_world,
            edge_points=edge_points,
            roi_bbox=(xmin, ymin, xmax, ymax),
            residual_error=residual,
            confidence=confidence,
            detection_method=arc_fit_method,
            source_type="FITTED",
        )
        self._store.add(mf)

        self._debug_data[feat.feature_id] = {
            "type": "arc",
            "roi": (xmin, ymin, xmax, ymax),
            "predicted_center": pixel_center,
            "predicted_radius": pixel_radius,
            "predicted_arc_points": arc_px,
            "edge_points": edge_points,
            "fitted_center": center,
            "fitted_radius": radius,
            "residual": residual,
            "confidence": confidence,
            "detection_method": arc_fit_method,
        }
        if arc_fit is not None:
            if "clusters" in arc_fit:
                self._debug_data[feat.feature_id]["radius_clusters"] = arc_fit["clusters"]
            if "candidate_edge_points" in arc_fit:
                self._debug_data[feat.feature_id]["candidate_edge_points"] = arc_fit[
                    "candidate_edge_points"
                ]
            if "all_contour_points" in arc_fit:
                self._debug_data[feat.feature_id]["all_contour_points"] = arc_fit[
                    "all_contour_points"
                ]
            if "component_fits" in arc_fit:
                self._debug_data[feat.feature_id]["component_fits"] = arc_fit[
                    "component_fits"
                ]

        dp = center - pixel_center
        _print(f"  Arc {feat.feature_id[:12]}: "
               f"predicted_r={pixel_radius:.1f}px fitted_r={radius:.1f}px "
               f"Δcenter=({dp[0]:.1f},{dp[1]:.1f}) "
               f"conf={confidence:.2f} pts={len(edge_points)}")
        return mf

    def _fit_arc_radius_from_dip_contours(
        self,
        center: np.ndarray,
        predicted_radius: float,
        arc_px: np.ndarray,
        roi_bbox: tuple[int, int, int, int],
    ) -> Optional[dict]:
        """Fit a target arc using QA-style DIPLib watershed contours + RANSAC."""
        if not HAS_DIP or self._image is None:
            return None

        xmin, ymin, xmax, ymax = roi_bbox
        crop = np.ascontiguousarray(self._image[ymin:ymax, xmin:xmax])
        if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
            return None

        try:
            dip_img = dip.Image(crop)
            dip_img.SetPixelSize([self._pixel_size_mm, self._pixel_size_mm], "mm")
            smoothed = dip.Gauss(dip_img, 0.4)
            grad = dip.Norm(dip.GradientMagnitude(smoothed))
            grad = dip.Opening(dip.Closing(grad, 3), 3)
            labels = np.array(dip.Watershed(
                grad,
                connectivity=1,
                maxDepth=3,
                flags={"correct", "labels"},
            )).astype(np.int32)
        except Exception:
            return None

        components = self._dip_watershed_contour_components(labels, xmin, ymin)
        if not components:
            return None
        contour_points = np.vstack(components).astype(np.float64)

        fits: list[dict] = []
        max_support = 1
        for component in components:
            near_arc = self._filter_points_near_polyline(
                component,
                arc_px,
                max(10.0, predicted_radius * 0.18),
                endpoint_margin_ratio=0.08,
            )
            if len(near_arc) < 8:
                continue

            radial = np.linalg.norm(near_arc - center, axis=1)
            band = (radial >= predicted_radius * 0.45) & (radial <= predicted_radius * 1.55)
            near_arc = near_arc[band]
            if len(near_arc) < 8:
                continue

            fit = self._ransac_circle_fit_with_prior(
                near_arc,
                center,
                predicted_radius,
                n_iterations=2500,
                threshold=2.2,
                min_inliers=8,
            )
            if fit is None:
                continue

            fit_center, radius, inliers, residual = fit
            support = near_arc[inliers]
            if len(support) < 8:
                continue
            max_support = max(max_support, len(support))
            fits.append({
                "center": fit_center,
                "radius": float(radius),
                "support": support,
                "candidate_points": near_arc,
                "residual": float(residual),
            })

        if not fits:
            return None

        max_center_shift = max(24.0, predicted_radius * 0.35)
        for item in fits:
            radius = float(item["radius"])
            fit_center = np.asarray(item["center"], dtype=np.float64)
            residual = float(item["residual"])
            support = item["support"]
            radius_score = max(0.0, 1.0 - abs(radius - predicted_radius) / max(predicted_radius * 0.28, 12.0))
            center_score = max(0.0, 1.0 - np.linalg.norm(fit_center - center) / max_center_shift)
            support_score = min(1.0, len(support) / max_support)
            residual_score = max(0.0, 1.0 - residual / 2.2)
            item["score"] = float(
                0.50 * radius_score
                + 0.20 * center_score
                + 0.20 * support_score
                + 0.10 * residual_score
            )

        selected = max(fits, key=lambda item: item["score"])
        fit_center = np.asarray(selected["center"], dtype=np.float64)
        radius = float(selected["radius"])
        support = selected["support"]
        residual = float(selected["residual"])

        if self._gradient is not None:
            h, w = self._gradient.shape[:2]
            ix = np.clip(np.round(support[:, 0]).astype(int), 0, w - 1)
            iy = np.clip(np.round(support[:, 1]).astype(int), 0, h - 1)
            gradient_strength = float(np.mean(self._gradient[iy, ix]))
        else:
            gradient_strength = 255.0

        component_debug = []
        for item in fits:
            component_debug.append({
                "center_px": [float(item["center"][0]), float(item["center"][1])],
                "radius_px": float(item["radius"]),
                "residual_px": float(item["residual"]),
                "support_count": int(len(item["support"])),
                "score": float(item["score"]),
                "selected": item is selected,
            })

        return {
            "method": "arc_dip_watershed_ransac",
            "center": fit_center,
            "radius": radius,
            "edge_points": support,
            "candidate_edge_points": selected["candidate_points"],
            "all_contour_points": contour_points,
            "component_fits": component_debug,
            "residual": residual,
            "gradient_strength": gradient_strength,
        }

    @staticmethod
    def _dip_watershed_contour_components(
        labels: np.ndarray,
        xmin: int,
        ymin: int,
    ) -> list[np.ndarray]:
        components: list[np.ndarray] = []
        for label in np.unique(labels):
            if int(label) == 0:
                continue
            mask = (labels == label).astype(np.uint8)
            if int(mask.sum()) < 20:
                continue
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                if len(contour) < 8:
                    continue
                pts = contour[:, 0, :].astype(np.float64)
                pts[:, 0] += xmin
                pts[:, 1] += ymin
                components.append(pts)
        return components

    @staticmethod
    def _dip_watershed_contour_points(
        labels: np.ndarray,
        xmin: int,
        ymin: int,
    ) -> np.ndarray:
        components = MeasurementPipeline._dip_watershed_contour_components(
            labels, xmin, ymin,
        )
        if not components:
            return np.empty((0, 2), dtype=np.float64)
        return np.vstack(components).astype(np.float64)

    @staticmethod
    def _solve_circle_from_3_points(
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
    ) -> Optional[tuple[np.ndarray, float]]:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        det = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if abs(det) < 1e-10:
            return None
        ux = ((x1 * x1 + y1 * y1) * (y2 - y3)
              + (x2 * x2 + y2 * y2) * (y3 - y1)
              + (x3 * x3 + y3 * y3) * (y1 - y2)) / det
        uy = ((x1 * x1 + y1 * y1) * (x3 - x2)
              + (x2 * x2 + y2 * y2) * (x1 - x3)
              + (x3 * x3 + y3 * y3) * (x2 - x1)) / det
        radius = float(np.hypot(x1 - ux, y1 - uy))
        return np.array([ux, uy], dtype=np.float64), radius

    @staticmethod
    def _fit_circle_least_squares(points: np.ndarray) -> Optional[tuple[np.ndarray, float]]:
        if len(points) < 3:
            return None
        x = points[:, 0]
        y = points[:, 1]
        z = x * x + y * y
        mat = np.column_stack([x, y, np.ones_like(x)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(mat, z, rcond=None)
        except np.linalg.LinAlgError:
            return None
        a, b, c = coeffs
        cx = float(a / 2.0)
        cy = float(b / 2.0)
        rad_sq = float(c + cx * cx + cy * cy)
        if rad_sq <= 0.0:
            return None
        return np.array([cx, cy], dtype=np.float64), float(np.sqrt(rad_sq))

    def _ransac_circle_fit_with_prior(
        self,
        points: np.ndarray,
        predicted_center: np.ndarray,
        predicted_radius: float,
        n_iterations: int = 2000,
        threshold: float = 2.0,
        min_inliers: int = 8,
    ) -> Optional[tuple[np.ndarray, float, np.ndarray, float]]:
        if len(points) < 3:
            return None

        rng = np.random.default_rng(12345)
        best: Optional[tuple[float, np.ndarray, float, np.ndarray, float]] = None
        max_center_shift = max(24.0, predicted_radius * 0.35)
        min_radius = predicted_radius * 0.65
        max_radius = predicted_radius * 1.35

        for _ in range(n_iterations):
            sample_idx = rng.choice(len(points), 3, replace=False)
            solved = self._solve_circle_from_3_points(
                points[sample_idx[0]], points[sample_idx[1]], points[sample_idx[2]],
            )
            if solved is None:
                continue
            center, radius = solved
            if not (min_radius <= radius <= max_radius):
                continue
            center_shift = float(np.linalg.norm(center - predicted_center))
            if center_shift > max_center_shift:
                continue

            distances = np.linalg.norm(points - center, axis=1)
            inliers = np.abs(distances - radius) <= threshold
            n_inliers = int(np.sum(inliers))
            if n_inliers < min_inliers:
                continue
            residual = float(np.mean(np.abs(distances[inliers] - radius)))
            score = (
                n_inliers
                - 2.5 * residual
                - 0.08 * center_shift
                - 0.03 * abs(radius - predicted_radius)
            )
            if best is None or score > best[0]:
                best = (float(score), center, float(radius), inliers, residual)

        if best is None:
            return None

        _, center, radius, inliers, _ = best
        inlier_points = points[inliers]
        refined = self._fit_circle_least_squares(inlier_points)
        if refined is not None:
            center, radius = refined
            distances = np.linalg.norm(points - center, axis=1)
            inliers = np.abs(distances - radius) <= threshold
            if int(np.sum(inliers)) >= min_inliers:
                inlier_points = points[inliers]
                refined2 = self._fit_circle_least_squares(inlier_points)
                if refined2 is not None:
                    center, radius = refined2

        center_shift = float(np.linalg.norm(center - predicted_center))
        if center_shift > max_center_shift * 1.15:
            return None
        if not (min_radius <= radius <= max_radius):
            return None

        distances = np.linalg.norm(points - center, axis=1)
        inliers = np.abs(distances - radius) <= threshold
        if int(np.sum(inliers)) < min_inliers:
            return None
        residual = float(np.mean(np.abs(distances[inliers] - radius)))
        return center.astype(np.float64), float(radius), inliers, residual

    def _fit_arc_radius_from_radial_edges(
        self,
        center: np.ndarray,
        predicted_radius: float,
        arc_px: np.ndarray,
    ) -> Optional[dict]:
        """Find arc radius from radial image-edge profiles along the CAD arc."""
        if self._gradient is None or len(arc_px) < 8:
            return None

        h, w = self._gradient.shape[:2]
        start = int(len(arc_px) * 0.08)
        end = max(start + 1, int(len(arc_px) * 0.92))
        samples = arc_px[start:end]
        search_width = max(14.0, predicted_radius * 0.28)
        r_start = max(1.0, predicted_radius - search_width)
        r_end = predicted_radius + search_width
        n_samples = max(21, int((r_end - r_start) * 2.0) + 1)
        sample_radii = np.linspace(r_start, r_end, n_samples)

        candidates: list[tuple[float, float, np.ndarray]] = []
        min_gradient = max(35.0, float(np.mean(self._gradient)) * 2.0)

        for arc_point in samples:
            direction = arc_point - center
            norm = float(np.linalg.norm(direction))
            if norm < 1.0:
                continue
            unit = direction / norm
            px = center[0] + sample_radii * unit[0]
            py = center[1] + sample_radii * unit[1]
            in_bounds = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            if not np.any(in_bounds):
                continue

            ix = np.clip(np.round(px).astype(int), 0, w - 1)
            iy = np.clip(np.round(py).astype(int), 0, h - 1)
            profile = self._gradient[iy, ix].astype(np.float64).copy()
            profile[~in_bounds] = 0.0
            peaks = self._arc_profile_peaks(profile, min_gradient)
            if not peaks:
                continue

            for peak in peaks:
                radius = self._subpixel_profile_radius(profile, sample_radii, peak)
                point = center + radius * unit
                candidates.append((float(radius), float(profile[peak]), point))

        if len(candidates) < 4:
            return None

        clusters = self._cluster_arc_radius_candidates(candidates, predicted_radius)
        if not clusters:
            return None

        selected = max(clusters, key=lambda item: item["score"])
        support = selected["items"]
        if len(support) < 4:
            return None

        radii = np.array([item[0] for item in support], dtype=np.float64)
        grads = np.array([item[1] for item in support], dtype=np.float64)
        points = np.array([item[2] for item in support], dtype=np.float64)
        radius = float(np.median(radii))
        residual = float(np.mean(np.abs(radii - radius)))
        gradient_strength = float(np.mean(grads))

        cluster_debug = []
        for cluster in clusters:
            values = np.array([item[0] for item in cluster["items"]], dtype=np.float64)
            cluster_debug.append({
                "count": len(cluster["items"]),
                "mean_radius_px": float(np.mean(values)),
                "median_radius_px": float(np.median(values)),
                "distance_from_predicted_px": float(abs(np.median(values) - predicted_radius)),
                "score": float(cluster["score"]),
                "selected": cluster is selected,
            })

        return {
            "method": "arc_radial_profile_inner",
            "radius": radius,
            "edge_points": points,
            "candidate_edge_points": np.array(
                [item[2] for item in candidates], dtype=np.float64,
            ),
            "residual": residual,
            "gradient_strength": gradient_strength,
            "clusters": cluster_debug,
        }

    @staticmethod
    def _arc_profile_peaks(
        profile: np.ndarray,
        min_gradient: float,
    ) -> list[int]:
        profile_mean = float(np.mean(profile))
        threshold = max(
            min_gradient,
            profile_mean * 2.5,
            float(np.percentile(profile, 75)) * 1.2,
        )
        peaks: list[int] = []
        n = len(profile)
        for idx in range(1, n - 1):
            if (profile[idx] >= threshold
                    and profile[idx] >= profile[idx - 1]
                    and profile[idx] >= profile[idx + 1]):
                peaks.append(idx)
        if n > 1 and profile[0] >= threshold and profile[0] >= profile[1]:
            peaks.append(0)
        if n > 1 and profile[-1] >= threshold and profile[-1] >= profile[-2]:
            peaks.append(n - 1)
        if not peaks:
            return []

        merged: list[int] = []
        run = [peaks[0]]
        for idx in peaks[1:]:
            if idx - run[-1] <= 2:
                run.append(idx)
            else:
                merged.append(max(run, key=lambda peak: profile[peak]))
                run = [idx]
        merged.append(max(run, key=lambda peak: profile[peak]))
        return merged

    @staticmethod
    def _subpixel_profile_radius(
        profile: np.ndarray,
        sample_radii: np.ndarray,
        peak: int,
    ) -> float:
        if not (1 <= peak <= len(profile) - 2):
            return float(sample_radii[peak])

        y_m1 = float(profile[peak - 1])
        y_0 = float(profile[peak])
        y_p1 = float(profile[peak + 1])
        denom = 2.0 * (2.0 * y_0 - y_m1 - y_p1)
        if abs(denom) <= 1e-10:
            return float(sample_radii[peak])

        offset = float(np.clip((y_p1 - y_m1) / denom, -0.5, 0.5))
        step = float(sample_radii[1] - sample_radii[0])
        return float(sample_radii[peak] + offset * step)

    @staticmethod
    def _cluster_arc_radius_candidates(
        candidates: list[tuple[float, float, np.ndarray]],
        predicted_radius: float,
    ) -> list[dict]:
        bin_width = 2.0
        bins: dict[int, list[tuple[float, float, np.ndarray]]] = {}
        for item in candidates:
            radius = item[0]
            key = int(round(radius / bin_width))
            bins.setdefault(key, []).append(item)

        if not bins:
            return []

        max_count = max(len(items) for items in bins.values())
        max_grad = max(float(np.mean([item[1] for item in items])) for items in bins.values())
        search_scale = max(predicted_radius * 0.28, 14.0)
        min_support = max(4, int(len(candidates) * 0.08))
        clusters: list[dict] = []

        for items in bins.values():
            if len(items) < min_support:
                continue
            radii = np.array([item[0] for item in items], dtype=np.float64)
            grads = np.array([item[1] for item in items], dtype=np.float64)
            radius = float(np.median(radii))
            closeness = max(0.0, 1.0 - abs(radius - predicted_radius) / search_scale)
            support_score = len(items) / max_count if max_count > 0 else 0.0
            grad_score = float(np.mean(grads)) / max_grad if max_grad > 0 else 0.0
            inner_score = 1.0 if radius <= predicted_radius else 0.0
            score = (
                0.65 * closeness * closeness
                + 0.20 * inner_score
                + 0.10 * grad_score
                + 0.05 * support_score
            )
            clusters.append({
                "items": items,
                "score": float(score),
            })

        return clusters

    @staticmethod
    def _sample_arc_world_points(geom: dict) -> np.ndarray:
        cx, cy, radius = geom["cx"], geom["cy"], geom["radius"]
        start_deg = float(geom.get("start_angle", 0.0))
        end_deg = float(geom.get("end_angle", 0.0))
        if end_deg < start_deg:
            end_deg += 360.0
        sweep = max(1.0, end_deg - start_deg)
        n = max(24, int(sweep / 3.0))
        angles = np.radians(np.linspace(start_deg, end_deg, n))
        return np.column_stack([
            cx + radius * np.cos(angles),
            cy + radius * np.sin(angles),
        ]).astype(np.float64)

    @staticmethod
    def _filter_points_near_polyline(
        points: np.ndarray,
        polyline: np.ndarray,
        max_distance: float,
        endpoint_margin_ratio: float = 0.0,
    ) -> np.ndarray:
        if len(points) == 0 or len(polyline) == 0:
            return np.empty((0, 2), dtype=np.float64)
        max_dist_sq = max_distance * max_distance
        keep = np.zeros(len(points), dtype=bool)
        start_index = int(len(polyline) * endpoint_margin_ratio)
        end_index = len(polyline) - start_index - 1
        chunk_size = 512
        for start in range(0, len(points), chunk_size):
            chunk = points[start:start + chunk_size]
            diff = chunk[:, None, :] - polyline[None, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            nearest_idx = np.argmin(dist_sq, axis=1)
            min_dist_sq = dist_sq[np.arange(len(chunk)), nearest_idx]
            near_curve = min_dist_sq <= max_dist_sq
            away_from_caps = (nearest_idx >= start_index) & (nearest_idx <= end_index)
            keep[start:start + chunk_size] = near_curve & away_from_caps
        return points[keep]

    def _measure_line(
        self,
        feat: CADFeature,
        paired_geometry: Optional[dict] = None,
        cache: bool = True,
    ) -> Optional[MeasuredFeature]:
        """Measure a line feature via perpendicular scanline sampling."""
        if self._line_engine is None:
            return None

        geom = feat.geometry
        roi_result = self._roi_predictor.predict_line_roi(geom, padding=50)
        if roi_result is None:
            return None
        roi, pixel_p1, pixel_p2 = roi_result

        preferred_side_point = None
        max_scan_width = None
        prefer_extreme_side = False
        lock_line_direction = False
        if paired_geometry is not None:
            paired_roi = self._roi_predictor.predict_line_roi(paired_geometry, padding=50)
            if paired_roi is not None:
                _, paired_p1, paired_p2 = paired_roi
                own_center = (pixel_p1 + pixel_p2) / 2.0
                paired_center = (paired_p1 + paired_p2) / 2.0
                line_vec = pixel_p2 - pixel_p1
                line_len = float(np.linalg.norm(line_vec))
                paired_vec = paired_p2 - paired_p1
                paired_len = float(np.linalg.norm(paired_vec))
                if line_len > 1e-6 and paired_len > 1e-6:
                    line_dir = line_vec / line_len
                    paired_dir = paired_vec / paired_len
                    # Pair-side fitting is only meaningful for near-parallel
                    # line-distance queries. Non-parallel line pairs keep the
                    # normal closest-edge behavior.
                    if abs(float(np.dot(line_dir, paired_dir))) > 0.95:
                        lock_line_direction = True
                        line_normal = np.array([-line_dir[1], line_dir[0]])
                        pair_gap_px = abs(float((paired_center - own_center) @ line_normal))
                        # Only close line pairs should force the inward
                        # edge. Distant line-distance queries are independent
                        # features; using the other line as a side target can
                        # pull the fit onto unrelated internal window edges.
                        if 1e-6 < pair_gap_px <= 250.0:
                            preferred_side_point = paired_center
                            max_scan_width = max(12.0, pair_gap_px * 0.45)
                            prefer_extreme_side = True

        # Fit — wide search to handle registration errors up to ~5mm
        result: Optional[LineFitResult] = self._line_engine.fit(
            pixel_p1, pixel_p2,
            scan_width=50.0,
            min_gradient=15.0,
            preferred_side_point=preferred_side_point,
            max_scan_width=max_scan_width,
            prefer_extreme_side=prefer_extreme_side,
            lock_direction=lock_line_direction,
        )
        if result is None and preferred_side_point is not None:
            result = self._line_engine.fit(
                pixel_p1, pixel_p2,
                scan_width=50.0,
                min_gradient=15.0,
                preferred_side_point=preferred_side_point,
                max_scan_width=None,
                prefer_extreme_side=prefer_extreme_side,
                lock_direction=True,
            )
        if result is None and preferred_side_point is not None:
            result = self._line_engine.fit(
                pixel_p1, pixel_p2,
                scan_width=50.0,
                min_gradient=15.0,
            )
        if result is None:
            _print(f"  Line {feat.feature_id[:12]}: NO EDGE FOUND "
                   f"(predicted=({pixel_p1[0]:.1f},{pixel_p1[1]:.1f})-"
                   f"({pixel_p2[0]:.1f},{pixel_p2[1]:.1f}))")
            return None

        dp1 = result.p1 - pixel_p1
        dp2 = result.p2 - pixel_p2
        displacement_px1 = float(np.linalg.norm(dp1))
        displacement_px2 = float(np.linalg.norm(dp2))
        _print(f"  Line {feat.feature_id[:12]}: "
               f"Δpx1=({dp1[0]:.1f},{dp1[1]:.1f}) "
               f"Δpx2=({dp2[0]:.1f},{dp2[1]:.1f}) "
               f"conf={result.confidence:.2f} pts={result.n_edge_points}")

        # Gradient quality validation: require strong edges
        image_grad_mean = float(np.mean(self._gradient)) if self._gradient is not None else 0.0
        if result.gradient_strength < max(45.0, image_grad_mean * 3.0):
            _print(f"  REJECTED: gradient_strength={result.gradient_strength:.1f} < threshold "
                   f"(edges too weak, likely noise)")
            return None

        # Convert fitted line endpoints to world coords
        # Apply residual distortion correction first
        pixel_pts = np.array([result.p1, result.p2])
        world_pts = self._pixel_points_to_world(pixel_pts)

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
            source_type="FITTED",
        )
        if cache:
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
            "pair_side_fit": paired_geometry is not None,
        }

        return mf

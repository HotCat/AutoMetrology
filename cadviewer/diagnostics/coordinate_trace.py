"""
Coordinate Trace System — traces every coordinate transformation in the measurement pipeline.

For each measured feature, logs all intermediate coordinates:
  CAD Feature → Registered Position → ROI Position → Fitted Edge Points →
  TPS Corrected Points → Final Measurement Geometry

This diagnostic reveals scale errors, coordinate system mixing, and double-correction bugs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np

from ..models.feature import CADFeature, FeatureType
from ..models.repository import FeatureRepository
from ..models.measured_feature import MeasuredFeature
from ..measurement.measurement_pipeline import MeasurementPipeline
from ..measurement.roi_predictor import FeatureROIPredictor
from ..registration import affine_solver
from ..calibration.residual_map import ResidualDistortionMap


@dataclass
class CoordinateTrace:
    """Trace of all coordinate transformations for a single feature."""

    feature_id: str
    cad_feature_id: str
    feature_type: str

    # Stage 1: CAD nominal geometry (mm)
    cad_geometry: Dict[str, float] = field(default_factory=dict)

    # Stage 2: CAD position projected to image via registration (pixels)
    registered_pixel: Dict[str, float] = field(default_factory=dict)

    # Stage 3: ROI region (pixels)
    roi_bbox: Dict[str, int] = field(default_factory=dict)

    # Stage 4: Fitted geometry in pixel space (before TPS)
    fitted_pixel: Dict[str, float] = field(default_factory=dict)

    # Stage 5: TPS correction applied (pixels)
    tps_correction: Dict[str, float] = field(default_factory=dict)  # (dx, dy) per point
    fitted_pixel_corrected: Dict[str, float] = field(default_factory=dict)

    # Stage 6: World coordinates via affine (mm)
    fitted_world: Dict[str, float] = field(default_factory=dict)

    # Stage 7: Final measurement (mm)
    final_measurement: Dict[str, float] = field(default_factory=dict)

    # Errors and diagnostics
    pixel_residual: float = 0.0
    tps_applied: bool = False
    affine_applied: bool = False

    # Additional context
    edge_point_count: int = 0
    confidence: float = 0.0
    source_type: str = "UNKNOWN"  # GeometrySourceType: FITTED, IMAGE_EDGE, MEASURED


@dataclass
class TraceReport:
    """Complete trace report for all features in a measurement."""

    traces: List[CoordinateTrace] = field(default_factory=list)
    pixel_size_mm: float = 0.0
    affine_params: Dict[str, float] = field(default_factory=dict)
    tps_built: bool = False
    tps_n_samples: int = 0
    image_size: tuple = (0, 0)

    def to_dict(self) -> dict:
        return {
            "traces": [asdict(t) for t in self.traces],
            "pixel_size_mm": self.pixel_size_mm,
            "affine_params": self.affine_params,
            "tps_built": self.tps_built,
            "tps_n_samples": self.tps_n_samples,
            "image_size": list(self.image_size),
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, default=_json_default))

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 70,
            "COORDINATE TRANSFORMATION TRACE REPORT",
            "=" * 70,
            f"Pixel Size: {self.pixel_size_mm:.6f} mm/px",
            f"Affine: scale={self.affine_params.get('scale_x', 0):.6f} "
            f"rot={self.affine_params.get('rotation_deg', 0):.4f}deg "
            f"tx={self.affine_params.get('tx', 0):.2f} ty={self.affine_params.get('ty', 0):.2f}",
            f"TPS Map: built={self.tps_built} samples={self.tps_n_samples}",
            f"Image Size: {self.image_size[0]}x{self.image_size[1]} px",
            "",
        ]

        for trace in self.traces:
            lines.extend([
                "-" * 70,
                f"Feature: {trace.feature_id} ({trace.feature_type})",
                f"CAD Feature ID: {trace.cad_feature_id}",
                f"Geometry Source: {trace.source_type}",
                "",
                "STAGE 1: CAD NOMINAL (mm)",
                f"  {trace.cad_geometry}",
                "",
                "STAGE 2: REGISTERED POSITION (pixels, via inv-affine)",
                f"  {trace.registered_pixel}",
                "",
                "STAGE 3: ROI BBOX (pixels)",
                f"  {trace.roi_bbox}",
                "",
                "STAGE 4: FITTED PIXEL (before TPS)",
                f"  {trace.fitted_pixel}",
                f"  Residual: {trace.pixel_residual:.4f} px",
                "",
                "STAGE 5: TPS CORRECTION",
                f"  Applied: {trace.tps_applied}",
                f"  Correction Vector: {trace.tps_correction}",
                f"  Corrected Pixel: {trace.fitted_pixel_corrected}",
                "",
                "STAGE 6: WORLD COORDINATES (mm, via affine)",
                f"  {trace.fitted_world}",
                "",
                "STAGE 7: FINAL MEASUREMENT",
                f"  {trace.final_measurement}",
                f"  Edge Points: {trace.edge_point_count}",
                f"  Confidence: {trace.confidence:.3f}",
                "",
            ])

        lines.append("=" * 70)
        return "\n".join(lines)


def _json_default(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class CoordinateTracer:
    """Traces coordinate transformations through the measurement pipeline."""

    def __init__(
        self,
        repo: FeatureRepository,
        affine: np.ndarray,
        pixel_size_mm: float,
        residual_map: Optional[ResidualDistortionMap] = None,
    ) -> None:
        self._repo = repo
        self._affine = affine
        self._pixel_size_mm = pixel_size_mm
        self._residual_map = residual_map
        self._roi_predictor = FeatureROIPredictor(affine)

    def trace_feature(
        self,
        cad_feature_id: str,
        fitted_geometry_pixel: Dict[str, float],
        fitted_geometry_world: Dict[str, float],
        roi_bbox: tuple,
        residual: float,
        confidence: float,
        edge_point_count: int,
        source_type: str = "FITTED",
    ) -> CoordinateTrace:
        """Create a complete trace for a single measured feature."""

        cad_feat = self._repo.get(cad_feature_id)
        if cad_feat is None:
            cad_feat = self._repo.get_by_handle(cad_feature_id)

        trace = CoordinateTrace(
            feature_id=f"{cad_feature_id}_trace",
            cad_feature_id=cad_feature_id,
            feature_type="unknown",
        )

        # Stage 1: CAD nominal geometry
        if cad_feat is not None:
            trace.feature_type = cad_feat.feature_type.name
            trace.cad_geometry = self._extract_cad_geometry(cad_feat)

            # Stage 2: Registered position (project CAD to pixel)
            trace.registered_pixel = self._project_cad_to_pixel(cad_feat)

        # Stage 3: ROI bbox
        trace.roi_bbox = {
            "xmin": roi_bbox[0],
            "ymin": roi_bbox[1],
            "xmax": roi_bbox[2],
            "ymax": roi_bbox[3],
        }

        # Stage 4: Fitted pixel geometry
        trace.fitted_pixel = fitted_geometry_pixel.copy()
        trace.pixel_residual = residual
        trace.confidence = confidence
        trace.edge_point_count = edge_point_count
        trace.source_type = source_type

        # Stage 5: TPS correction
        trace.tps_applied = self._residual_map is not None and self._residual_map.is_built
        if trace.tps_applied and fitted_geometry_pixel:
            trace.fitted_pixel_corrected, trace.tps_correction = \
                self._apply_tps_correction(fitted_geometry_pixel)

        # Stage 6: World coordinates via affine
        trace.fitted_world = fitted_geometry_world.copy()
        trace.affine_applied = True

        # Stage 7: Final measurement
        trace.final_measurement = self._compute_final_measurement(trace)

        return trace

    def _extract_cad_geometry(self, feat: CADFeature) -> Dict[str, float]:
        """Extract CAD geometry as a flat dict."""
        g = feat.geometry
        if feat.feature_type == FeatureType.CIRCLE:
            return {"cx": g["cx"], "cy": g["cy"], "radius": g["radius"]}
        elif feat.feature_type == FeatureType.LINE:
            return {"x1": g["x1"], "y1": g["y1"], "x2": g["x2"], "y2": g["y2"]}
        return {}

    def _project_cad_to_pixel(self, feat: CADFeature) -> Dict[str, float]:
        """Project CAD geometry to pixel coordinates via inverse affine."""
        g = feat.geometry
        if feat.feature_type == FeatureType.CIRCLE:
            cx, cy = g["cx"], g["cy"]
            pixel_pt = self._roi_predictor.project_point(np.array([cx, cy]))
            return {"cx_px": pixel_pt[0], "cy_px": pixel_pt[1]}
        elif feat.feature_type == FeatureType.LINE:
            x1, y1, x2, y2 = g["x1"], g["y1"], g["x2"], g["y2"]
            p1 = self._roi_predictor.project_point(np.array([x1, y1]))
            p2 = self._roi_predictor.project_point(np.array([x2, y2]))
            return {"x1_px": p1[0], "y1_px": p1[1], "x2_px": p2[0], "y2_px": p2[1]}
        return {}

    def _apply_tps_correction(
        self, fitted_pixel: Dict[str, float]
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        """Apply TPS correction to fitted pixel coordinates."""
        corrected = fitted_pixel.copy()
        correction = {}

        if self._residual_map is None or not self._residual_map.is_built:
            return corrected, correction

        # Extract points from fitted_geometry based on feature type
        points_to_correct = []
        point_names = []

        if "cx" in fitted_pixel and "cy" in fitted_pixel:
            points_to_correct.append([fitted_pixel["cx"], fitted_pixel["cy"]])
            point_names.append("center")
        if "x1" in fitted_pixel and "y1" in fitted_pixel:
            points_to_correct.append([fitted_pixel["x1"], fitted_pixel["y1"]])
            point_names.append("p1")
        if "x2" in fitted_pixel and "y2" in fitted_pixel:
            points_to_correct.append([fitted_pixel["x2"], fitted_pixel["y2"]])
            point_names.append("p2")

        if points_to_correct:
            pts_arr = np.array(points_to_correct)
            corrections_arr = self._residual_map.correction_vectors(pts_arr)

            for i, name in enumerate(point_names):
                dx, dy = corrections_arr[i]
                correction[f"{name}_dx"] = dx
                correction[f"{name}_dy"] = dy

            corrected_pts = self._residual_map.correct(pts_arr)

            for i, name in enumerate(point_names):
                if name == "center":
                    corrected["cx"] = corrected_pts[i, 0]
                    corrected["cy"] = corrected_pts[i, 1]
                elif name == "p1":
                    corrected["x1"] = corrected_pts[i, 0]
                    corrected["y1"] = corrected_pts[i, 1]
                elif name == "p2":
                    corrected["x2"] = corrected_pts[i, 0]
                    corrected["y2"] = corrected_pts[i, 1]

        return corrected, correction

    def _compute_final_measurement(self, trace: CoordinateTrace) -> Dict[str, float]:
        """Compute final measurement values from the trace."""
        result = {}

        if trace.feature_type == "CIRCLE":
            if trace.fitted_world:
                result["radius_mm"] = trace.fitted_world.get("radius", 0)
                result["cx_mm"] = trace.fitted_world.get("cx", 0)
                result["cy_mm"] = trace.fitted_world.get("cy", 0)

                # Compare to CAD nominal
                if trace.cad_geometry:
                    nominal_r = trace.cad_geometry.get("radius", 0)
                    if nominal_r > 0:
                        result["radius_deviation_mm"] = result["radius_mm"] - nominal_r
                        result["radius_deviation_pct"] = \
                            (result["radius_mm"] - nominal_r) / nominal_r * 100

        elif trace.feature_type == "LINE":
            if trace.fitted_world:
                x1 = trace.fitted_world.get("x1", 0)
                y1 = trace.fitted_world.get("y1", 0)
                x2 = trace.fitted_world.get("x2", 0)
                y2 = trace.fitted_world.get("y2", 0)
                length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                result["length_mm"] = length

                # Compare to CAD nominal length
                if trace.cad_geometry:
                    nx1 = trace.cad_geometry.get("x1", 0)
                    ny1 = trace.cad_geometry.get("y1", 0)
                    nx2 = trace.cad_geometry.get("x2", 0)
                    ny2 = trace.cad_geometry.get("y2", 0)
                    nominal_length = math.sqrt((nx2 - nx1) ** 2 + (ny2 - ny1) ** 2)
                    if nominal_length > 0:
                        result["length_deviation_mm"] = length - nominal_length
                        result["length_deviation_pct"] = \
                            (length - nominal_length) / nominal_length * 100

        return result

    def trace_measurement(
        self,
        measured_features: List[MeasuredFeature],
    ) -> TraceReport:
        """Create a complete trace report from a list of measured features."""

        report = TraceReport(
            pixel_size_mm=self._pixel_size_mm,
            affine_params=affine_solver.extract_params(self._affine),
            tps_built=self._residual_map is not None and self._residual_map.is_built,
            tps_n_samples=self._residual_map.n_samples if self._residual_map else 0,
            image_size=self._residual_map.image_size if self._residual_map else (0, 0),
        )

        for mf in measured_features:
            trace = self.trace_feature(
                cad_feature_id=mf.cad_feature_id,
                fitted_geometry_pixel=mf.fitted_geometry,
                fitted_geometry_world=mf.fitted_geometry_world,
                roi_bbox=mf.roi_bbox,
                residual=mf.residual_error,
                confidence=mf.confidence,
                edge_point_count=len(mf.edge_points) if mf.edge_points is not None else 0,
                source_type=mf.source_type,
            )
            report.traces.append(trace)

        return report


def run_coordinate_trace(
    repo: FeatureRepository,
    affine: np.ndarray,
    pixel_size_mm: float,
    measured_features: List[MeasuredFeature],
    residual_map: Optional[ResidualDistortionMap] = None,
    output_path: Optional[Path] = None,
) -> TraceReport:
    """Run a coordinate trace analysis and optionally save the report."""

    tracer = CoordinateTracer(repo, affine, pixel_size_mm, residual_map)
    report = tracer.trace_measurement(measured_features)

    if output_path is not None:
        report.save(output_path)
        print(f"[TRACE] Report saved to {output_path}")

    return report
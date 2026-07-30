"""Dual-light metrology pipeline.

Backlight frames are used only to fit the hollow window and solve CAD pose.
Ring-light frames are used for printed-line fitting.  Metric scale comes only
from the configured chessboard pixel size.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from ..calibration.residual_map import ResidualDistortionMap
from ..measurement.measurement_pipeline import MeasurementPipeline
from ..measurement.query_parser import QueryParser
from ..measurement.window_dimensions import (
    WindowLineDimensions,
    measure_window_line_dimensions,
    side_line_corners,
)
from ..models.feature import CADFeature, FeatureType
from ..models.measured_feature import MeasuredFeature, MeasuredFeatureStore
from ..models.query import QueryResult
from ..models.repository import FeatureRepository
from ..registration import affine_solver
from ..registration.window_line_registration import (
    _cad_corners_from_line_features,
    _corner_aspect,
    _resolve_line,
    register_window_lines,
)
from .evaluator import QueryEvaluator

try:
    import cv2
    HAS_CV2 = True
except ImportError:  # pragma: no cover
    HAS_CV2 = False

from tools.validate_backlight_hollow_window import _fit_window, _to_gray


@dataclass
class DualLightArtifacts:
    backlight_raw_image_path: str = ""
    ring_light_raw_image_path: str = ""
    backlight_overlay_path: str = ""
    ring_light_overlay_path: str = ""
    json_path: str = ""


@dataclass
class DualLightMeasurementResult:
    results: list[QueryResult]
    pipeline: "DualLightMeasurementPipeline"
    registration: dict
    artifacts: DualLightArtifacts = field(default_factory=DualLightArtifacts)


class DualLightMeasurementPipeline:
    """Measurement pipeline facade for dual-light query evaluation."""

    def __init__(
        self,
        repo: FeatureRepository,
        backlight_image: np.ndarray,
        ring_light_image: np.ndarray,
        edge_tokens: list[str],
        pixel_size_mm: float,
        residual_map: Optional[ResidualDistortionMap] = None,
        line_pair_bias_mode: str = "center",
        line_fit_side_mode: str = "auto",
        line_fit_side_overrides: Optional[dict[str, str]] = None,
        fit_mode: str = "light-inner",
        light_fraction: float = 0.95,
        edge_bias: Optional[str] = None,
        prefer_diplib: bool = True,
    ) -> None:
        if not HAS_CV2:
            raise RuntimeError("OpenCV is required for dual-light metrology")
        if pixel_size_mm <= 0:
            raise ValueError("pixel_size_mm must be positive")
        if backlight_image is None or ring_light_image is None:
            raise ValueError("backlight_image and ring_light_image are required")
        if backlight_image.shape[:2] != ring_light_image.shape[:2]:
            raise ValueError(
                "Backlight and ring-light images must have identical resolution/ROI"
            )
        if len(edge_tokens) != 4:
            raise ValueError("Dual-light registration requires exactly 4 CAD window edges")

        self._repo = repo
        self._pixel_size_mm = float(pixel_size_mm)
        self._backlight_bgr = _ensure_bgr(backlight_image)
        self._ring_bgr = _ensure_bgr(ring_light_image)
        self._backlight_gray = _to_gray(self._backlight_bgr)
        self._ring_gray = _to_gray(self._ring_bgr)
        self._point_corrector = (
            residual_map.correct
            if residual_map is not None and getattr(residual_map, "is_built", False)
            else None
        )
        self._window_features = [_resolve_line(repo, token) for token in edge_tokens]
        self._cad_corners = _cad_corners_from_line_features(self._window_features)

        self._backlight_fit = _fit_window(
            self._backlight_gray,
            pixel_size_mm=self._pixel_size_mm,
            gt_width_mm=0.0,
            gt_height_mm=0.0,
            prefer_diplib=prefer_diplib,
            fit_mode=fit_mode,
            edge_bias=(
                edge_bias
                if edge_bias is not None
                else ("strongest" if str(fit_mode).lower() == "gradient" else "inner")
            ),
            light_fraction=light_fraction,
            undistorted=True,
            target_aspect=_corner_aspect(self._cad_corners),
        )
        self._image_corners = side_line_corners(
            self._backlight_fit.side_lines,
            point_corrector=self._point_corrector,
        )
        self._backlight_dimensions = measure_window_line_dimensions(
            self._backlight_fit.side_lines,
            pixel_size_mm=self._pixel_size_mm,
            point_corrector=self._point_corrector,
        )
        self._transform = _solve_fixed_scale_transform(
            self._image_corners,
            self._cad_corners,
            self._pixel_size_mm,
        )
        self._window_measured = _measured_window_features(
            self._window_features,
            self._cad_corners,
            self._image_corners,
            self._transform,
        )
        self._store = MeasuredFeatureStore()
        for mf in self._window_measured.values():
            self._store.add(mf)

        self._ring_pipeline = MeasurementPipeline(
            repo,
            self._ring_gray,
            self._transform,
            pixel_size_mm=self._pixel_size_mm,
            residual_map=residual_map,
            pixel_to_world_transform=self._transform,
            line_pair_bias_mode=line_pair_bias_mode,
            line_fit_side_mode=line_fit_side_mode,
            line_fit_side_overrides=line_fit_side_overrides,
        )
        self._debug_data = self._build_initial_debug_data()

    @property
    def store(self) -> MeasuredFeatureStore:
        return self._store

    @property
    def measurement_transform(self) -> np.ndarray:
        return self._transform

    @property
    def backlight_image(self) -> np.ndarray:
        return self._backlight_bgr

    @property
    def ring_light_image(self) -> np.ndarray:
        return self._ring_bgr

    @property
    def registration_debug(self) -> dict:
        return self._registration_debug()

    def validate_ring_pose_consistency(self, max_mean_corner_delta_px: float = 25.0) -> dict:
        """Check that ring-light and backlight frames show the same window pose."""
        diagnostic = {
            "status": "not_checked",
            "max_mean_corner_delta_px": float(max_mean_corner_delta_px),
        }
        try:
            result = register_window_lines(
                self._repo,
                self._ring_bgr,
                edge_tokens=[f.feature_id for f in self._window_features],
                pixel_size_mm=self._pixel_size_mm,
                prefer_homography=False,
                detection_mode="dark",
            )
        except Exception as exc:
            diagnostic.update({"status": "unavailable", "error": str(exc)})
            return diagnostic

        try:
            ring_corners = np.asarray(result.image_corners, dtype=np.float64)
            backlight_corners = np.asarray(self._image_corners, dtype=np.float64)
            deltas = np.linalg.norm(ring_corners - backlight_corners, axis=1)
            mean_delta = float(np.mean(deltas))
            diagnostic.update({
                "status": "ok" if mean_delta <= max_mean_corner_delta_px else "mismatch",
                "mean_corner_delta_px": mean_delta,
                "corner_deltas_px": [float(v) for v in deltas],
                "ring_window_confidence": float(result.confidence),
                "ring_component_bbox": list(result.component_bbox),
            })
            if mean_delta > max_mean_corner_delta_px:
                raise RuntimeError(
                    "Backlight and ring-light window poses disagree "
                    f"(mean corner delta {mean_delta:.1f}px). "
                    "Rejecting dual-light measurement because the two frames "
                    "may not be the same product pose."
                )
        except RuntimeError:
            raise
        except Exception as exc:
            diagnostic.update({"status": "unavailable", "error": str(exc)})
        return diagnostic

    def validate_orientation_with_ring_prints(
        self,
        query_text: str,
        line_fit_side_overrides: Optional[dict[str, str]] = None,
        ambiguity_margin: float = 0.15,
    ) -> dict:
        """Reject 180-degree ambiguity when ring-light witnesses are symmetric.

        The hollow window alone cannot distinguish a product loaded normally
        from one rotated 180 degrees.  We compare the current corner mapping
        with the 180-degree alternate using printed-line fit quality.  If both
        hypotheses are similarly good, the measurement is unsafe because CAD
        line labels can be swapped silently.
        """
        window_ids = {feature.feature_id for feature in self._window_features}
        printed_ids = _printed_line_ids_from_queries(self._repo, query_text, window_ids)
        diagnostic = {
            "status": "not_checked",
            "printed_line_count": len(printed_ids),
            "ambiguity_margin": float(ambiguity_margin),
            "candidates": [],
        }
        if not printed_ids:
            diagnostic["status"] = "no_printed_line_witness"
            return diagnostic

        candidates = []
        for roll in (0, 2):
            image_corners = np.roll(self._image_corners, -roll, axis=0)
            transform = _solve_fixed_scale_transform(
                image_corners,
                self._cad_corners,
                self._pixel_size_mm,
            )
            score, details = _printed_line_pose_score(
                self._repo,
                self._ring_gray,
                transform,
                self._pixel_size_mm,
                printed_ids,
                line_fit_side_overrides or {},
            )
            candidates.append({
                "corner_roll": int(roll),
                "score": float(score),
                "details": details,
            })
        candidates.sort(key=lambda item: item["score"])
        diagnostic["candidates"] = candidates
        best = candidates[0]["score"]
        alternate = candidates[1]["score"]
        rel_gap = (alternate - best) / max(abs(best), 1.0)
        diagnostic["relative_score_gap"] = float(rel_gap)
        if rel_gap <= ambiguity_margin:
            diagnostic["status"] = "ambiguous_180_warning"
            diagnostic["warning"] = (
                "The backlight hollow window is 180-degree symmetric, and the "
                "configured ring-light printed-line witnesses do not distinguish "
                "the normal and 180-degree hypotheses strongly enough. Continuing "
                "with the configured corner order; use an asymmetric witness or "
                "mechanical loading keying if flipped products must be rejected."
            )
            return diagnostic
        if candidates[0]["corner_roll"] != 0:
            diagnostic["status"] = "wrong_orientation"
            raise RuntimeError(
                "Dual-light registration selected the 180-degree alternate as "
                "a better fit. Product orientation likely does not match the "
                "configured CAD orientation; measurement aborted."
            )
        diagnostic["status"] = "ok"
        return diagnostic

    def measure_feature(self, cad_feature_id: str) -> Optional[MeasuredFeature]:
        if cad_feature_id in self._window_measured:
            return self._window_measured[cad_feature_id]
        mf = self._ring_pipeline.measure_feature(cad_feature_id)
        if mf is not None:
            self._store.add(mf)
        return mf

    def measure_line_pair(
        self, cad_feature_id_1: str, cad_feature_id_2: str,
    ) -> tuple[Optional[MeasuredFeature], Optional[MeasuredFeature]]:
        if cad_feature_id_1 in self._window_measured or cad_feature_id_2 in self._window_measured:
            return self.measure_feature(cad_feature_id_1), self.measure_feature(cad_feature_id_2)
        mf1, mf2 = self._ring_pipeline.measure_line_pair(cad_feature_id_1, cad_feature_id_2)
        for mf in (mf1, mf2):
            if mf is not None:
                self._store.add(mf)
        return mf1, mf2

    def get_debug_data(self) -> dict:
        data = dict(self._debug_data)
        data.update(self._ring_pipeline.get_debug_data())
        return data

    def save_artifacts(
        self,
        output_dir: Path | str,
        results: Optional[list[QueryResult]] = None,
        prefix: str = "dual_light",
        metadata: Optional[dict] = None,
    ) -> DualLightArtifacts:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base = f"{prefix}_{stamp}"
        raw_backlight = output / f"{base}_backlight.png"
        raw_ring = output / f"{base}_ring.png"
        backlight_overlay = output / f"{base}_backlight_overlay.png"
        ring_overlay = output / f"{base}_ring_overlay.png"
        json_path = output / f"{base}.json"
        cv2.imwrite(str(raw_backlight), self._backlight_bgr)
        cv2.imwrite(str(raw_ring), self._ring_bgr)
        self.save_backlight_overlay(backlight_overlay)
        self.save_ring_overlay(ring_overlay)
        payload = self.to_json_dict(results=results, metadata=metadata)
        payload["artifacts"] = {
            "backlight_raw_image_path": str(raw_backlight),
            "ring_light_raw_image_path": str(raw_ring),
            "backlight_overlay_path": str(backlight_overlay),
            "ring_light_overlay_path": str(ring_overlay),
        }
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return DualLightArtifacts(
            backlight_raw_image_path=str(raw_backlight),
            ring_light_raw_image_path=str(raw_ring),
            backlight_overlay_path=str(backlight_overlay),
            ring_light_overlay_path=str(ring_overlay),
            json_path=str(json_path),
        )

    def save_backlight_overlay(self, path: Path | str) -> None:
        canvas = self._backlight_bgr.copy()
        for mf in self._window_measured.values():
            gp = mf.fitted_geometry
            _draw_line(
                canvas,
                np.array([gp["x1"], gp["y1"]], dtype=np.float64),
                np.array([gp["x2"], gp["y2"]], dtype=np.float64),
                (0, 255, 0),
                3,
            )
        for feature in self._window_features:
            pts = _project_cad_line_to_pixel(self._transform, feature)
            _draw_line(canvas, pts[0], pts[1], (255, 0, 0), 2)
            _label(canvas, _feature_token(feature), np.mean(pts, axis=0), (255, 0, 0))
        dims = self._backlight_dimensions
        _label(
            canvas,
            (
                f"W {dims.opposite_line_width_mm:.4f}mm "
                f"({dims.opposite_line_width_px:.2f}px)"
            ),
            np.array([40.0, 70.0], dtype=np.float64),
            (0, 255, 0),
        )
        _label(
            canvas,
            (
                f"H {dims.opposite_line_height_mm:.4f}mm "
                f"({dims.opposite_line_height_px:.2f}px)"
            ),
            np.array([40.0, 115.0], dtype=np.float64),
            (0, 255, 0),
        )
        _label(
            canvas,
            (
                "diag corner "
                f"W={dims.corner_segment_width_mm:.4f}mm "
                f"H={dims.corner_segment_height_mm:.4f}mm"
            ),
            np.array([40.0, 160.0], dtype=np.float64),
            (255, 255, 0),
        )
        _label(
            canvas,
            (
                f"angle dW={dims.width_stats.angle_difference_deg:.4f}deg "
                f"dH={dims.height_stats.angle_difference_deg:.4f}deg; "
                f"range W={dims.width_stats.separation_range_px:.3f}px "
                f"H={dims.height_stats.separation_range_px:.3f}px"
            ),
            np.array([40.0, 205.0], dtype=np.float64),
            (255, 255, 0),
        )
        _write_resized(Path(path), canvas)

    def save_ring_overlay(self, path: Path | str) -> None:
        canvas = self._ring_bgr.copy()
        feature_ids = set(self._store._by_cad_id.keys())  # debug/export only
        for fid in feature_ids:
            feature = self._repo.get(fid)
            if feature is None or feature.feature_type != FeatureType.LINE:
                continue
            pts = _project_cad_line_to_pixel(self._transform, feature)
            _draw_line(canvas, pts[0], pts[1], (255, 0, 0), 2)
            _label(canvas, _feature_token(feature), np.mean(pts, axis=0), (255, 0, 0))
        for mf in self._store.all_measured():
            if mf.feature_type != FeatureType.LINE:
                continue
            gp = mf.fitted_geometry
            source = "backlight" if mf.cad_feature_id in self._window_measured else "ring"
            color = (0, 255, 255) if source == "backlight" else (0, 255, 0)
            _draw_line(
                canvas,
                np.array([gp["x1"], gp["y1"]], dtype=np.float64),
                np.array([gp["x2"], gp["y2"]], dtype=np.float64),
                color,
                3,
            )
        _write_resized(Path(path), canvas)

    def to_json_dict(
        self,
        results: Optional[list[QueryResult]] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        output = self._registration_debug()
        output["metadata"] = metadata or {}
        output["fitted_lines"] = {
            mf.cad_feature_id: _measured_feature_to_dict(mf)
            for mf in self._store.all_measured()
        }
        output["measurements"] = [
            _query_result_to_dict(result) for result in (results or [])
        ]
        if results is not None:
            output["geometric_consistency"] = self.geometric_consistency_diagnostics(results)
        return output

    def geometric_consistency_diagnostics(self, results: list[QueryResult]) -> dict:
        """Report gap + window size consistency against printed contour span."""
        window_ids = set(self._window_measured.keys())
        center = np.mean(self._image_corners, axis=0)
        gaps: dict[str, dict] = {}
        printed_by_axis: dict[str, set[str]] = {"x": set(), "y": set()}

        for result in results:
            inst = result.instruction
            if inst is None or result.value is None:
                continue
            feature_1 = _resolve_query_feature(self._repo, inst.feature_id_1)
            feature_2 = _resolve_query_feature(self._repo, inst.feature_id_2)
            if feature_1 is None or feature_2 is None:
                continue
            ids = [feature_1.feature_id, feature_2.feature_id]
            in_window = [fid in window_ids for fid in ids]
            if in_window.count(True) != 1:
                continue
            window_feature = feature_1 if in_window[0] else feature_2
            printed_feature = feature_2 if in_window[0] else feature_1
            side = _window_side_from_measured(
                self._window_measured[window_feature.feature_id],
                center,
            )
            if side is None:
                continue
            axis = "x" if side in ("left", "right") else "y"
            gaps[side] = {
                "query": inst.raw_text,
                "gap_mm": float(result.value),
                "window_feature": _feature_token(window_feature),
                "printed_feature": _feature_token(printed_feature),
                "axis": axis,
            }
            printed_by_axis[axis].add(printed_feature.feature_id)

        diagnostics = {
            "horizontal": _axis_consistency(
                self._store,
                printed_by_axis["x"],
                gaps.get("left"),
                gaps.get("right"),
                self._backlight_dimensions.opposite_line_width_mm,
                axis="x",
            ),
            "vertical": _axis_consistency(
                self._store,
                printed_by_axis["y"],
                gaps.get("top"),
                gaps.get("bottom"),
                self._backlight_dimensions.opposite_line_height_mm,
                axis="y",
            ),
            "note": (
                "Diagnostic only. Measurements are not modified to force "
                "gap + window + gap consistency."
            ),
        }
        return diagnostics

    def _build_initial_debug_data(self) -> dict:
        data: dict = {}
        for fid, mf in self._window_measured.items():
            gp = mf.fitted_geometry
            data[fid] = {
                "type": "line",
                "source_image": "backlight",
                "edge_points": mf.edge_points,
                "fitted_line": np.array([[gp["x1"], gp["y1"]], [gp["x2"], gp["y2"]]]),
                "residual": mf.residual_error,
                "confidence": mf.confidence,
                "detection_method": mf.detection_method,
            }
        return data

    def _registration_debug(self) -> dict:
        return {
            "mode": "dual_light_fixed_scale",
            "pixel_size_mm": self._pixel_size_mm,
            "scale_source": "chessboard pixel_size_mm only",
            "forbidden_scale_source": "hollow window scale diagnostic only",
            "window_edges": [_feature_token(f) for f in self._window_features],
            "backlight_window_fit": {
                "fit_mode": self._backlight_fit.fit_mode,
                "light_fraction": self._backlight_fit.light_fraction,
                "gradient_method": self._backlight_fit.gradient_method,
                "component_bbox": list(self._backlight_fit.bbox),
                "corners_px": self._image_corners.tolist(),
                "line_equations_ax_by_c": {
                    key: [float(v) for v in value]
                    for key, value in self._backlight_fit.side_lines.items()
                },
                "window_dimensions": self._backlight_dimensions.to_dict(
                    self._pixel_size_mm,
                ),
                "fit_quality": self._backlight_fit.fit_quality,
                "authoritative_distances_px": self._backlight_dimensions.distances_px(),
                "authoritative_distances_with_chessboard_scale_mm": (
                    self._backlight_dimensions.distances_mm()
                ),
                "corner_segment_distances_diagnostic_px": {
                    "corner_segment_width_px": (
                        self._backlight_dimensions.corner_segment_width_px
                    ),
                    "corner_segment_height_px": (
                        self._backlight_dimensions.corner_segment_height_px
                    ),
                },
                "raw_distances_px": self._backlight_dimensions.distances_px(),
                "raw_distances_with_chessboard_scale_mm": (
                    self._backlight_dimensions.distances_mm()
                ),
            },
            "raw_window_derived_scale_diagnostic_only": _raw_window_scale_diagnostics(
                self._cad_corners,
                self._image_corners,
                self._pixel_size_mm,
            ),
            "fixed_scale_pose": _params_from_fixed_transform(
                self._transform,
                self._pixel_size_mm,
            ),
            "measurement_transform_pixel_to_cad": self._transform.tolist(),
            "backlight_measured_window_lines": {
                fid: _measured_feature_to_dict(mf)
                for fid, mf in self._window_measured.items()
            },
        }


def run_dual_light_measurement(
    *,
    repo: FeatureRepository,
    query_text: str,
    backlight_image: np.ndarray,
    ring_light_image: np.ndarray,
    edge_tokens: list[str],
    pixel_size_mm: float,
    residual_map: Optional[ResidualDistortionMap] = None,
    line_pair_bias_mode: str = "center",
    line_fit_side_mode: str = "auto",
    line_fit_side_overrides: Optional[dict[str, str]] = None,
    output_dir: Optional[Path | str] = None,
    metadata: Optional[dict] = None,
    fit_mode: str = "light-inner",
    light_fraction: float = 0.95,
    edge_bias: Optional[str] = None,
    orientation_guard_enabled: bool = True,
) -> DualLightMeasurementResult:
    pipeline = DualLightMeasurementPipeline(
        repo,
        backlight_image,
        ring_light_image,
        edge_tokens=edge_tokens,
        pixel_size_mm=pixel_size_mm,
        residual_map=residual_map,
        line_pair_bias_mode=line_pair_bias_mode,
        line_fit_side_mode=line_fit_side_mode,
        line_fit_side_overrides=line_fit_side_overrides,
        fit_mode=fit_mode,
        light_fraction=light_fraction,
        edge_bias=edge_bias,
    )
    if orientation_guard_enabled:
        ring_pose_diagnostic = pipeline.validate_ring_pose_consistency()
        orientation_diagnostic = pipeline.validate_orientation_with_ring_prints(
            query_text,
            line_fit_side_overrides=line_fit_side_overrides,
        )
    else:
        ring_pose_diagnostic = {
            "status": "skipped",
            "reason": "dual-light orientation guard disabled",
        }
        orientation_diagnostic = {
            "status": "skipped",
            "reason": "dual-light orientation guard disabled",
        }
    if orientation_diagnostic.get("status") != "ok" and orientation_guard_enabled:
        status = orientation_diagnostic.get("status", "unresolved")
        warning = orientation_diagnostic.get("warning") or orientation_diagnostic.get("error") or ""
        message = (
            "Dual-light orientation could not be resolved: "
            f"{status}. "
            f"{warning} "
            "The ring-light witnesses were insufficient to disambiguate the "
            "backlight window orientation. Use an asymmetric witness or "
            "manual window registration."
        ).strip()
        raise RuntimeError(message)
    evaluator = QueryEvaluator(repo, measurement_pipeline=pipeline)
    results = evaluator.evaluate(query_text)
    artifacts = DualLightArtifacts()
    if output_dir is not None:
        artifacts = pipeline.save_artifacts(
            output_dir,
            results=results,
            metadata=metadata,
        )
    registration = pipeline.registration_debug
    registration["ring_pose_consistency"] = ring_pose_diagnostic
    registration["orientation_validation"] = orientation_diagnostic
    registration["orientation_guard_enabled"] = bool(orientation_guard_enabled)
    if orientation_diagnostic.get("status") != "ok" and not orientation_guard_enabled:
        registration["orientation_guard_warning"] = (
            "Dual-light orientation guard is disabled; continuing with the "
            "configured backlight corner order."
        )
    registration["geometric_consistency"] = pipeline.geometric_consistency_diagnostics(
        results,
    )
    registration["artifacts"] = artifacts.__dict__.copy()
    return DualLightMeasurementResult(
        results=results,
        pipeline=pipeline,
        registration=registration,
        artifacts=artifacts,
    )


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    return image.copy()


def _line_points_world(geom: dict) -> np.ndarray:
    return np.array(
        [[float(geom["x1"]), float(geom["y1"])],
         [float(geom["x2"]), float(geom["y2"])]],
        dtype=np.float64,
    )


def _feature_token(feature: CADFeature) -> str:
    return str(feature.dxf_handle or feature.feature_id[:8])


def _resolve_query_feature(repo: FeatureRepository, raw_id: str) -> Optional[CADFeature]:
    feature = repo.get(raw_id) or repo.get_by_handle(raw_id)
    if feature is not None:
        return feature
    needle = str(raw_id or "").lower()
    for candidate in repo.all_features():
        handle = str(candidate.dxf_handle or "").lower()
        if candidate.feature_id.lower().startswith(needle):
            return candidate
        if handle and handle.startswith(needle):
            return candidate
    return None


def _printed_line_ids_from_queries(
    repo: FeatureRepository,
    query_text: str,
    window_ids: set[str],
) -> list[str]:
    ids: list[str] = []
    try:
        instructions = QueryParser().parse(query_text)
    except Exception:
        return ids
    for inst in instructions:
        for raw_id in (inst.feature_id_1, inst.feature_id_2):
            feature = _resolve_query_feature(repo, raw_id)
            if feature is None:
                continue
            if feature.feature_type != FeatureType.LINE:
                continue
            if feature.feature_id in window_ids:
                continue
            if feature.feature_id not in ids:
                ids.append(feature.feature_id)
    return ids


def _printed_line_pose_score(
    repo: FeatureRepository,
    ring_gray: np.ndarray,
    transform: np.ndarray,
    pixel_size_mm: float,
    printed_ids: list[str],
    line_fit_side_overrides: dict[str, str],
) -> tuple[float, list[dict]]:
    pipeline = MeasurementPipeline(
        repo,
        ring_gray,
        transform,
        pixel_size_mm=pixel_size_mm,
        pixel_to_world_transform=transform,
        line_fit_side_overrides=line_fit_side_overrides,
    )
    details = []
    total = 0.0
    for feature_id in printed_ids:
        feature = repo.get(feature_id)
        measured = pipeline.measure_feature(feature_id)
        if measured is None:
            score = 9999.0
            details.append({
                "feature": _feature_token(feature) if feature is not None else feature_id,
                "status": "no_measurement",
                "score": score,
            })
            total += score
            continue
        debug = pipeline.get_debug_data().get(feature_id, {})
        try:
            disp1 = float(np.linalg.norm(
                np.asarray(debug.get("fitted_p1"), dtype=np.float64)
                - np.asarray(debug.get("predicted_p1"), dtype=np.float64)
            ))
            disp2 = float(np.linalg.norm(
                np.asarray(debug.get("fitted_p2"), dtype=np.float64)
                - np.asarray(debug.get("predicted_p2"), dtype=np.float64)
            ))
            displacement = (disp1 + disp2) * 0.5
        except Exception:
            displacement = 9999.0
        score = (
            float(measured.residual_error)
            + 0.05 * displacement
            + 10.0 * max(0.0, 1.0 - float(measured.confidence))
        )
        details.append({
            "feature": _feature_token(feature) if feature is not None else feature_id,
            "status": "ok",
            "score": float(score),
            "residual_px": float(measured.residual_error),
            "confidence": float(measured.confidence),
            "predicted_to_fitted_displacement_px": float(displacement),
        })
        total += score
    return total / max(len(printed_ids), 1), details


def _pixel_metric(points_px: np.ndarray, pixel_size_mm: float) -> np.ndarray:
    pts = np.asarray(points_px, dtype=np.float64)
    return np.column_stack([pts[:, 0] * pixel_size_mm, -pts[:, 1] * pixel_size_mm])


def _solve_fixed_scale_transform(
    image_corners_px: np.ndarray,
    cad_corners: np.ndarray,
    pixel_size_mm: float,
) -> np.ndarray:
    image_metric = _pixel_metric(image_corners_px, pixel_size_mm)
    metric_to_cad = affine_solver.solve_rigid_with_fixed_scale(
        image_metric,
        np.asarray(cad_corners, dtype=np.float64),
        scale=1.0,
    )
    pixel_to_metric = np.array([
        [pixel_size_mm, 0.0, 0.0],
        [0.0, -pixel_size_mm, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return metric_to_cad @ pixel_to_metric


def _transform_points(transform: np.ndarray, points_px: np.ndarray) -> np.ndarray:
    return affine_solver.apply_projective(transform, np.asarray(points_px, dtype=np.float64))


def _cad_segment_feature(
    features: list[CADFeature],
    a: np.ndarray,
    b: np.ndarray,
) -> CADFeature:
    best = None
    best_err = float("inf")
    seg_vec = b - a
    seg_len = float(np.linalg.norm(seg_vec))
    if seg_len <= 1e-12:
        raise RuntimeError("Degenerate ordered CAD window segment")
    seg_dir = seg_vec / seg_len
    seg_mid = (a + b) * 0.5
    for feature in features:
        pts = _line_points_world(feature.geometry)
        endpoint_err = min(
            float(np.linalg.norm(pts[0] - a) + np.linalg.norm(pts[1] - b)),
            float(np.linalg.norm(pts[1] - a) + np.linalg.norm(pts[0] - b)),
        )
        vec = pts[1] - pts[0]
        length = float(np.linalg.norm(vec))
        if length <= 1e-12:
            continue
        direction = vec / length
        parallel_err = 1.0 - abs(float(np.dot(seg_dir, direction)))
        normal = np.array([-seg_dir[1], seg_dir[0]], dtype=np.float64)
        offset_err = abs(float((np.mean(pts, axis=0) - seg_mid) @ normal))
        err = min(endpoint_err, offset_err * 10.0 + parallel_err * 1000.0)
        if err < best_err:
            best = feature
            best_err = err
    if best is None or best_err > 5.0:
        raise RuntimeError("Could not map ordered CAD window segment to selected CAD line")
    return best


def _measured_window_features(
    window_features: list[CADFeature],
    cad_corners: np.ndarray,
    image_corners: np.ndarray,
    transform: np.ndarray,
) -> dict[str, MeasuredFeature]:
    measured: dict[str, MeasuredFeature] = {}
    for idx in range(4):
        cad_a = cad_corners[idx]
        cad_b = cad_corners[(idx + 1) % 4]
        px_a = image_corners[idx]
        px_b = image_corners[(idx + 1) % 4]
        feature = _cad_segment_feature(window_features, cad_a, cad_b)
        world = _transform_points(transform, np.vstack([px_a, px_b]))
        edge_points = np.vstack([px_a, px_b]).astype(np.float64)
        mf = MeasuredFeature(
            feature_id=str(uuid.uuid4()),
            cad_feature_id=feature.feature_id,
            feature_type=FeatureType.LINE,
            fitted_geometry={
                "x1": float(px_a[0]),
                "y1": float(px_a[1]),
                "x2": float(px_b[0]),
                "y2": float(px_b[1]),
            },
            fitted_geometry_world={
                "x1": float(world[0, 0]),
                "y1": float(world[0, 1]),
                "x2": float(world[1, 0]),
                "y2": float(world[1, 1]),
            },
            edge_points=edge_points,
            roi_bbox=_bbox_for_points(edge_points),
            residual_error=0.0,
            confidence=1.0,
            detection_method="backlight_hollow_window",
            source_type="FITTED",
        )
        measured[feature.feature_id] = mf
    return measured


def _bbox_for_points(points: np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(points, dtype=np.float64)
    return (
        int(math.floor(float(np.min(arr[:, 0])))),
        int(math.floor(float(np.min(arr[:, 1])))),
        int(math.ceil(float(np.max(arr[:, 0])))),
        int(math.ceil(float(np.max(arr[:, 1])))),
    )


def _raw_window_scale_diagnostics(
    cad_corners: np.ndarray,
    image_corners: np.ndarray,
    pixel_size_mm: float,
) -> dict:
    cad_lengths = [
        float(np.linalg.norm(cad_corners[(idx + 1) % 4] - cad_corners[idx]))
        for idx in range(4)
    ]
    px_lengths = [
        float(np.linalg.norm(image_corners[(idx + 1) % 4] - image_corners[idx]))
        for idx in range(4)
    ]
    scale_estimates = [
        cad / px for cad, px in zip(cad_lengths, px_lengths) if px > 1e-12
    ]
    mean = float(np.mean(scale_estimates)) if scale_estimates else None
    mismatch = None
    if mean is not None and pixel_size_mm > 0:
        mismatch = float((mean - pixel_size_mm) / pixel_size_mm)
    return {
        "cad_segment_lengths_mm": cad_lengths,
        "fitted_segment_lengths_px": px_lengths,
        "fitted_segment_lengths_with_chessboard_scale_mm": [
            px * pixel_size_mm for px in px_lengths
        ],
        "cad_over_pixel_scale_estimates_mm_per_px": scale_estimates,
        "mean_window_derived_scale_mm_per_px": mean,
        "chessboard_pixel_size_mm": pixel_size_mm,
        "relative_mismatch_vs_chessboard": mismatch,
        "warning": (
            "Window-derived scale differs from chessboard scale; diagnostic only."
            if mismatch is not None and abs(mismatch) > 0.01 else ""
        ),
    }


def _project_cad_line_to_pixel(transform: np.ndarray, feature: CADFeature) -> np.ndarray:
    inv = affine_solver.invert(transform)
    return affine_solver.apply_projective(inv, _line_points_world(feature.geometry))


def _draw_line(canvas: np.ndarray, p1: np.ndarray, p2: np.ndarray, color, width: int = 2) -> None:
    cv2.line(
        canvas,
        tuple(np.round(p1).astype(int)),
        tuple(np.round(p2).astype(int)),
        color,
        width,
        cv2.LINE_AA,
    )


def _label(canvas: np.ndarray, text: str, point: np.ndarray, color=(255, 255, 255)) -> None:
    xy = tuple(np.round(point).astype(int))
    cv2.putText(canvas, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def _write_resized(path: Path, image: np.ndarray) -> None:
    h, w = image.shape[:2]
    scale = min(1.0, 2200.0 / max(h, w))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    cv2.imwrite(str(path), image)


def _params_from_fixed_transform(transform: np.ndarray, pixel_size_mm: float) -> dict:
    pixel_to_metric = np.array([
        [pixel_size_mm, 0.0, 0.0],
        [0.0, -pixel_size_mm, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    metric_to_cad = transform @ np.linalg.inv(pixel_to_metric)
    return {
        "rotation_deg": float(np.degrees(np.arctan2(metric_to_cad[1, 0], metric_to_cad[0, 0]))),
        "translation_mm": [float(metric_to_cad[0, 2]), float(metric_to_cad[1, 2])],
        "fixed_scale": 1.0,
    }


def _measured_feature_to_dict(mf: MeasuredFeature) -> dict:
    return {
        "feature_id": mf.cad_feature_id,
        "source": (
            "backlight_window" if mf.detection_method == "backlight_hollow_window"
            else "ring_light_printed"
        ),
        "feature_type": mf.feature_type.name,
        "pixel_geometry": mf.fitted_geometry,
        "world_geometry": mf.fitted_geometry_world,
        "residual_px": float(mf.residual_error),
        "confidence": float(mf.confidence),
        "detection_method": mf.detection_method,
    }


def _window_side_from_measured(
    mf: MeasuredFeature,
    center_px: np.ndarray,
) -> Optional[str]:
    geom = mf.fitted_geometry
    p1 = np.array([geom["x1"], geom["y1"]], dtype=np.float64)
    p2 = np.array([geom["x2"], geom["y2"]], dtype=np.float64)
    vec = p2 - p1
    mid = (p1 + p2) * 0.5
    if abs(vec[1]) >= abs(vec[0]):
        return "left" if mid[0] < center_px[0] else "right"
    return "top" if mid[1] < center_px[1] else "bottom"


def _axis_consistency(
    store: MeasuredFeatureStore,
    printed_ids: set[str],
    gap_a: Optional[dict],
    gap_b: Optional[dict],
    window_size_mm: float,
    *,
    axis: str,
) -> dict:
    diagnostic = {
        "status": "not_available",
        "axis": axis,
        "window_size_mm": float(window_size_mm),
        "gap_a": gap_a,
        "gap_b": gap_b,
    }
    if gap_a is None or gap_b is None or len(printed_ids) < 2:
        return diagnostic
    measured = [
        store.get_by_cad_id(feature_id)
        for feature_id in printed_ids
    ]
    measured = [mf for mf in measured if mf is not None and mf.feature_type == FeatureType.LINE]
    if len(measured) < 2:
        return diagnostic
    best_pair = None
    best_span = None
    for i, first in enumerate(measured):
        for second in measured[i + 1:]:
            span = _line_distance_world_features(first, second)
            if best_span is None or span > best_span:
                best_span = span
                best_pair = (first, second)
    if best_pair is None or best_span is None:
        return diagnostic
    gap_sum = float(gap_a["gap_mm"]) + float(window_size_mm) + float(gap_b["gap_mm"])
    discrepancy = gap_sum - float(best_span)
    diagnostic.update({
        "status": "ok",
        "gap_plus_window_plus_gap_mm": gap_sum,
        "printed_contour_span_mm": float(best_span),
        "discrepancy_mm": float(discrepancy),
        "printed_features": [
            best_pair[0].cad_feature_id,
            best_pair[1].cad_feature_id,
        ],
    })
    return diagnostic


def _line_distance_world_features(first: MeasuredFeature, second: MeasuredFeature) -> float:
    g1 = first.fitted_geometry_world
    g2 = second.fitted_geometry_world
    p1 = np.array([g1["x1"], g1["y1"]], dtype=np.float64)
    p2 = np.array([g1["x2"], g1["y2"]], dtype=np.float64)
    q1 = np.array([g2["x1"], g2["y1"]], dtype=np.float64)
    q2 = np.array([g2["x2"], g2["y2"]], dtype=np.float64)
    vec = p2 - p1
    length = float(np.linalg.norm(vec))
    if length <= 1e-12:
        return 0.0
    normal = np.array([-vec[1] / length, vec[0] / length], dtype=np.float64)
    return abs(float(((q1 - p1) @ normal + (q2 - p1) @ normal) * 0.5))


def _query_result_to_dict(result: QueryResult) -> dict:
    inst = result.instruction
    return {
        "query": inst.raw_text if inst else "",
        "status": result.status,
        "value": result.value,
        "nominal": result.nominal,
        "deviation": result.deviation,
        "tolerance_abs": result.tolerance_abs,
        "geometry_source": result.geometry_source,
        "error_message": result.error_message,
        "feature_geometry_audit": result.feature_geometry_audit or {},
    }

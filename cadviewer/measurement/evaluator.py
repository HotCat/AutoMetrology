"""
QueryEvaluator — parse and evaluate measurement queries.

DATA CONTRACT:
  The `value` field in QueryResult is ALWAYS derived from image-fitted
  MeasuredFeature geometry. CAD geometry provides ONLY the nominal reference.

  If image fitting fails, the query returns status="no_measurement" with
  value=None. The system NEVER substitutes CAD nominal as the measured value.

Geometry flow per query:
  1. Parse query → resolve feature IDs
  2. Trigger image-based fitting for each feature (via MeasurementPipeline)
  3. Nominal: computed from CADFeature.geometry (reference only)
  4. Measured: computed from MeasuredFeature.fitted_geometry_world (image-derived)
  5. Audit: log exact geometry values used for each feature
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import numpy as np

from ..models.query import QueryInstruction, QueryResult, QueryType
from ..models.repository import FeatureRepository
from ..models.measured_feature import MeasuredFeature, MeasuredFeatureStore
from .query_parser import QueryParser
from .measurement_pipeline import MeasurementPipeline

logger = logging.getLogger(__name__)

_AUDIT = logging.getLogger("meas.audit")


def _audit(msg: str) -> None:
    _AUDIT.info(msg)
    logger.info(f"[AUDIT] {msg}")


class QueryEvaluator:
    """Parse and evaluate measurement queries using fitted image geometry."""

    def __init__(
        self,
        repo: FeatureRepository,
        measurement_pipeline: Optional[MeasurementPipeline] = None,
    ) -> None:
        self._repo = repo
        self._parser = QueryParser()
        self._pipeline = measurement_pipeline

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
        """Evaluate a single measurement query instruction."""
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

        _audit(f"Query: {inst.raw_text}")
        _audit(f"  Resolved IDs: {fid1[:12]}..., {fid2[:12]}...")

        # Compute nominal from CAD geometry (reference value only)
        if inst.query_type == QueryType.CIRCLE_DISTANCE:
            nominal = self._nominal_circle_distance(fid1, fid2)
        elif inst.query_type == QueryType.LINE_DISTANCE:
            nominal = self._nominal_line_distance(fid1, fid2)
        else:
            return QueryResult(
                instruction=inst, status="error",
                error_message="Unsupported query type",
            )

        if nominal is None:
            return QueryResult(
                instruction=inst, status="error",
                error_message="Nominal computation failed",
            )

        _audit(f"  Nominal (CAD): {nominal:.4f} mm")

        # Compute measured from image-fitted geometry
        measured, audit_data = self._measured_dimension_with_audit(
            inst.query_type, fid1, fid2,
        )

        if measured is not None:
            deviation = round(measured - nominal, 4)
            _audit(f"  Measured (IMAGE): {measured:.4f} mm")
            _audit(f"  Deviation: {deviation:.4f} mm")
            _audit(f"  Source: MEASURED (image-fitted geometry)")
            return QueryResult(
                instruction=inst,
                value=round(measured, 4),
                status="ok",
                nominal=round(nominal, 4),
                deviation=deviation,
                geometry_source="MEASURED",
                feature_geometry_audit=audit_data,
            )

        # Image fitting failed — do NOT substitute CAD nominal
        _audit(f"  FAILED: No image measurement available")
        _audit(f"  Source: NONE (fitting failed or unavailable)")
        return QueryResult(
            instruction=inst,
            value=None,
            status="no_measurement",
            error_message=(
                "Image fitting failed — no measured geometry available. "
                "Possible causes: poor registration, low contrast, "
                "feature outside ROI, or no measurement pipeline."
            ),
            nominal=round(nominal, 4),
            deviation=None,
            geometry_source="NONE",
            feature_geometry_audit=audit_data,
        )

    # ── Measured dimension computation with audit trail ───────────

    def _measured_dimension_with_audit(
        self, query_type: QueryType, fid1: str, fid2: str,
    ) -> tuple[Optional[float], dict]:
        """Compute measured dimension and return full audit trail."""
        audit: Dict = {
            "fid1": fid1[:12],
            "fid2": fid2[:12],
            "pipeline_available": self._pipeline is not None,
        }

        if self._pipeline is None:
            audit["failure_reason"] = "no_measurement_pipeline"
            return None, audit

        # Ensure both features are measured
        mf1 = self._pipeline.measure_feature(fid1)
        mf2 = self._pipeline.measure_feature(fid2)

        if mf1 is None:
            audit["failure_reason"] = f"feature1_not_measured (id={fid1[:12]})"
            _audit(f"  Feature 1 ({fid1[:12]}): fitting returned None")
            return None, audit
        if mf2 is None:
            audit["failure_reason"] = f"feature2_not_measured (id={fid2[:12]})"
            _audit(f"  Feature 2 ({fid2[:12]}): fitting returned None")
            return None, audit

        # Log per-feature geometry details
        self._audit_feature("Feature 1", fid1, mf1, audit)
        self._audit_feature("Feature 2", fid2, mf2, audit)

        if not mf1.is_valid():
            audit["failure_reason"] = (
                f"feature1_invalid "
                f"(conf={mf1.confidence:.2f}, resid={mf1.residual_error:.2f})"
            )
            _audit(
                f"  Feature 1 INVALID: "
                f"confidence={mf1.confidence:.2f} (need >0.2), "
                f"residual={mf1.residual_error:.2f} (need <5.0)"
            )
            return None, audit
        if not mf2.is_valid():
            audit["failure_reason"] = (
                f"feature2_invalid "
                f"(conf={mf2.confidence:.2f}, resid={mf2.residual_error:.2f})"
            )
            _audit(
                f"  Feature 2 INVALID: "
                f"confidence={mf2.confidence:.2f} (need >0.2), "
                f"residual={mf2.residual_error:.2f} (need <5.0)"
            )
            return None, audit

        # Both features are valid — compute from image-fitted geometry
        if query_type == QueryType.CIRCLE_DISTANCE:
            result = self._measured_circle_distance(mf1, mf2)
        elif query_type == QueryType.LINE_DISTANCE:
            result = self._measured_line_distance(mf1, mf2)
        else:
            audit["failure_reason"] = "unsupported_query_type"
            return None, audit

        audit["measured_value"] = result
        return result, audit

    def _audit_feature(
        self, label: str, fid: str, mf: MeasuredFeature, audit: dict,
    ) -> None:
        """Log full geometry audit for one measured feature."""
        cad_geom = self._get_cad_geometry(fid)
        source_type = getattr(mf, 'source_type', 'IMAGE_EDGE')

        _audit(f"  {label} ({fid[:12]}):")
        _audit(f"    source_type: {source_type}")
        _audit(f"    detection_method: {mf.detection_method}")
        _audit(f"    confidence: {mf.confidence:.3f}")
        _audit(f"    residual_error: {mf.residual_error:.3f} px")
        _audit(f"    edge_points_count: {len(mf.edge_points)}")
        _audit(f"    roi_bbox: {mf.roi_bbox}")

        if cad_geom is not None:
            _audit(f"    CAD geometry: {cad_geom}")

        _audit(f"    Fitted (pixel): {mf.fitted_geometry}")
        _audit(f"    Fitted (world): {mf.fitted_geometry_world}")

        # Store in audit dict
        key = label.lower().replace(" ", "_")
        audit[key] = {
            "source_type": source_type,
            "cad_geometry": cad_geom,
            "fitted_geometry_px": mf.fitted_geometry,
            "fitted_geometry_world": mf.fitted_geometry_world,
            "confidence": mf.confidence,
            "residual_error": mf.residual_error,
            "edge_points_count": len(mf.edge_points),
            "roi_bbox": mf.roi_bbox,
            "is_valid": mf.is_valid(),
        }

    # ── Measured dimension computation ────────────────────────────

    def _measured_dimension(
        self, query_type: QueryType, fid1: str, fid2: str,
    ) -> Optional[float]:
        """Compute dimension from MeasuredFeature fitted geometry."""
        result, _ = self._measured_dimension_with_audit(query_type, fid1, fid2)
        return result

    def _measured_circle_distance(
        self, mf1: MeasuredFeature, mf2: MeasuredFeature,
    ) -> float:
        """Center-to-center distance from fitted circle geometry."""
        g1 = mf1.fitted_geometry_world
        g2 = mf2.fitted_geometry_world
        # Assertion: geometry must come from image fitting, not CAD
        assert mf1.source_type in ("IMAGE_EDGE", "FITTED", "MEASURED"), (
            f"Data contract violation: Feature 1 source_type={mf1.source_type}, "
            f"expected IMAGE_EDGE, FITTED, or MEASURED"
        )
        assert mf2.source_type in ("IMAGE_EDGE", "FITTED", "MEASURED"), (
            f"Data contract violation: Feature 2 source_type={mf2.source_type}, "
            f"expected IMAGE_EDGE, FITTED, or MEASURED"
        )
        dx = g2["cx"] - g1["cx"]
        dy = g2["cy"] - g1["cy"]
        return math.sqrt(dx * dx + dy * dy)

    def _measured_line_distance(
        self, mf1: MeasuredFeature, mf2: MeasuredFeature,
    ) -> float:
        """Perpendicular distance from fitted line geometry."""
        g1 = mf1.fitted_geometry_world
        g2 = mf2.fitted_geometry_world
        assert mf1.source_type in ("IMAGE_EDGE", "FITTED", "MEASURED"), (
            f"Data contract violation: Feature 1 source_type={mf1.source_type}"
        )
        assert mf2.source_type in ("IMAGE_EDGE", "FITTED", "MEASURED"), (
            f"Data contract violation: Feature 2 source_type={mf2.source_type}"
        )
        x1, y1 = g1["x1"], g1["y1"]
        x2, y2 = g1["x2"], g1["y2"]
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-12:
            return 0.0
        nx, ny = -dy / length, dx / length
        lx1, ly1 = g2["x1"], g2["y1"]
        lx2, ly2 = g2["x2"], g2["y2"]
        d1 = abs((lx1 - x1) * nx + (ly1 - y1) * ny)
        d2 = abs((lx2 - x1) * nx + (ly2 - y1) * ny)
        return (d1 + d2) / 2

    # ── Nominal (CAD) dimension computation ───────────────────────

    def _nominal_circle_distance(self, fid1: str, fid2: str) -> Optional[float]:
        g1 = self._get_cad_geometry(fid1)
        g2 = self._get_cad_geometry(fid2)
        if g1 is None or g2 is None:
            return None
        dx = g2.get("cx", 0) - g1.get("cx", 0)
        dy = g2.get("cy", 0) - g1.get("cy", 0)
        return math.sqrt(dx * dx + dy * dy)

    def _nominal_line_distance(self, fid1: str, fid2: str) -> Optional[float]:
        g1 = self._get_cad_geometry(fid1)
        g2 = self._get_cad_geometry(fid2)
        if g1 is None or g2 is None:
            return None
        x1, y1 = g1.get("x1", 0), g1.get("y1", 0)
        x2, y2 = g1.get("x2", 0), g1.get("y2", 0)
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
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
        if self._repo.get(raw_id):
            return raw_id
        feat = self._repo.get_by_handle(raw_id)
        if feat:
            return feat.feature_id
        for feat in self._repo.all_features():
            if feat.feature_id.startswith(raw_id):
                return feat.feature_id
        return None

    def _get_cad_geometry(self, feature_id: str) -> Optional[dict]:
        feat = self._repo.get(feature_id)
        return feat.geometry if feat else None

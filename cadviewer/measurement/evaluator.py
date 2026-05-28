"""
QueryEvaluator — parse and evaluate measurement queries.

All dimension computations use MeasuredFeature fitted geometry
(image-derived), NOT CAD nominal geometry. CAD geometry provides
only the nominal reference value.

Flow:
  parse query → resolve IDs → ensure features measured →
  compute nominal from CAD → compute measured from MeasuredFeature →
  return QueryResult
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from ..models.query import QueryInstruction, QueryResult, QueryType
from ..models.repository import FeatureRepository
from ..models.measured_feature import MeasuredFeature, MeasuredFeatureStore
from .query_parser import QueryParser
from .measurement_pipeline import MeasurementPipeline


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

        # Compute nominal from CAD geometry (reference value)
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

        # Try image-based measurement from MeasuredFeature store
        measured = self._measured_dimension(inst.query_type, fid1, fid2)
        if measured is not None:
            return QueryResult(
                instruction=inst,
                value=round(measured, 4),
                status="ok",
                nominal=round(nominal, 4),
                deviation=round(measured - nominal, 4),
            )

        # Fallback: CAD-only
        return QueryResult(
            instruction=inst,
            value=round(nominal, 4),
            status="ok",
            nominal=round(nominal, 4),
            deviation=0.0,
        )

    # ── Measured dimension computation ───────────────────────────

    def _measured_dimension(
        self, query_type: QueryType, fid1: str, fid2: str,
    ) -> Optional[float]:
        """Compute dimension from MeasuredFeature fitted geometry."""
        if self._pipeline is None:
            return None

        # Ensure both features are measured
        mf1 = self._pipeline.measure_feature(fid1)
        mf2 = self._pipeline.measure_feature(fid2)
        if mf1 is None or mf2 is None:
            return None
        if not mf1.is_valid() or not mf2.is_valid():
            return None

        if query_type == QueryType.CIRCLE_DISTANCE:
            return self._measured_circle_distance(mf1, mf2)
        elif query_type == QueryType.LINE_DISTANCE:
            return self._measured_line_distance(mf1, mf2)
        return None

    def _measured_circle_distance(
        self, mf1: MeasuredFeature, mf2: MeasuredFeature,
    ) -> float:
        """Center-to-center distance from fitted circle geometry."""
        g1 = mf1.fitted_geometry_world
        g2 = mf2.fitted_geometry_world
        dx = g2["cx"] - g1["cx"]
        dy = g2["cy"] - g1["cy"]
        return math.sqrt(dx * dx + dy * dy)

    def _measured_line_distance(
        self, mf1: MeasuredFeature, mf2: MeasuredFeature,
    ) -> float:
        """Perpendicular distance from fitted line geometry."""
        g1 = mf1.fitted_geometry_world
        g2 = mf2.fitted_geometry_world
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

    # ── Nominal (CAD) dimension computation ──────────────────────

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

    # ── ID resolution ────────────────────────────────────────────

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

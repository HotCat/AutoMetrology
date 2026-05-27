"""
QueryEvaluator — orchestrates query execution pipeline.

Parse → Resolve IDs → Measure → Build results.

Supports two modes:
  - CAD-only: measures purely from CAD feature geometry (no image needed)
  - Full: uses image correspondences for measured vs nominal comparison
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from ..models.query import QueryInstruction, QueryResult, QueryType
from ..models.repository import FeatureRepository
from .query_parser import QueryParser


class QueryEvaluator:
    """Parse and evaluate measurement queries."""

    def __init__(
        self,
        repo: FeatureRepository,
    ) -> None:
        self._repo = repo
        self._parser = QueryParser()

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
        """Evaluate a single query instruction using CAD geometry."""
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

        # Measure from CAD geometry
        if inst.query_type == QueryType.CIRCLE_DISTANCE:
            value = self._measure_cad_circle_distance(fid1, fid2)
        elif inst.query_type == QueryType.LINE_DISTANCE:
            value = self._measure_cad_line_distance(fid1, fid2)
        else:
            value = None

        if value is None:
            return QueryResult(
                instruction=inst, status="error",
                error_message="Measurement computation failed",
            )

        return QueryResult(
            instruction=inst,
            value=value,
            status="ok",
            nominal=value,
            deviation=0.0,
        )

    # ── CAD-only measurement functions ────────────────────────────

    def _measure_cad_circle_distance(self, fid1: str, fid2: str) -> Optional[float]:
        """Center-to-center distance between two CAD circles."""
        g1 = self._get_cad_geometry(fid1)
        g2 = self._get_cad_geometry(fid2)
        if g1 is None or g2 is None:
            return None
        cx1, cy1 = g1.get("cx", 0), g1.get("cy", 0)
        cx2, cy2 = g2.get("cx", 0), g2.get("cy", 0)
        return math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)

    def _measure_cad_line_distance(self, fid1: str, fid2: str) -> Optional[float]:
        """Perpendicular distance between two CAD lines."""
        g1 = self._get_cad_geometry(fid1)
        g2 = self._get_cad_geometry(fid2)
        if g1 is None or g2 is None:
            return None
        x1, y1 = g1.get("x1", 0), g1.get("y1", 0)
        x2, y2 = g1.get("x2", 0), g1.get("y2", 0)
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx ** 2 + dy ** 2)
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
        # 1. Direct feature_id match
        if self._repo.get(raw_id):
            return raw_id
        # 2. DXF handle match
        feat = self._repo.get_by_handle(raw_id)
        if feat:
            return feat.feature_id
        # 3. Partial UUID match
        for feat in self._repo.all_features():
            if feat.feature_id.startswith(raw_id):
                return feat.feature_id
        return None

    def _get_cad_geometry(self, feature_id: str) -> Optional[dict]:
        feat = self._repo.get(feature_id)
        return feat.geometry if feat else None

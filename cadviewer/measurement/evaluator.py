"""
QueryEvaluator — orchestrates query execution pipeline.

Parse → Resolve IDs → Check correspondences → Measure → Build results.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..models.query import QueryInstruction, QueryResult, QueryType
from ..models.repository import FeatureRepository
from ..models.correspondence import CorrespondenceMap
from ..models.image_feature import ImageFeatureRepository
from .query_parser import QueryParser
from .engine import MeasurementEngine


class QueryEvaluator:
    """Parse and evaluate measurement queries."""

    def __init__(
        self,
        repo: FeatureRepository,
        corr_map: CorrespondenceMap,
        image_repo: ImageFeatureRepository,
        affine: np.ndarray,
        pixel_size_mm: float,
    ) -> None:
        self._repo = repo
        self._parser = QueryParser()
        self._engine = MeasurementEngine(corr_map, image_repo, affine, pixel_size_mm)

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
            result = self._evaluate_instruction(inst)
            results.append(result)

        return results

    def _evaluate_instruction(self, inst: QueryInstruction) -> QueryResult:
        """Evaluate a single query instruction."""
        # Resolve feature IDs
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

        # Check correspondences exist
        corr1 = self._engine._corr_map.get_for_cad(fid1)
        corr2 = self._engine._corr_map.get_for_cad(fid2)
        if not corr1:
            return QueryResult(
                instruction=inst, status="no_correspondence",
                error_message=f"No correspondence for {inst.feature_id_1}",
            )
        if not corr2:
            return QueryResult(
                instruction=inst, status="no_correspondence",
                error_message=f"No correspondence for {inst.feature_id_2}",
            )

        # Measure
        if inst.query_type == QueryType.CIRCLE_DISTANCE:
            value = self._engine.measure_circle_distance(fid1, fid2)
        elif inst.query_type == QueryType.LINE_DISTANCE:
            value = self._engine.measure_line_distance(fid1, fid2)
        else:
            value = None

        if value is None:
            return QueryResult(
                instruction=inst, status="error",
                error_message="Measurement computation failed",
            )

        # Compute nominal and deviation
        nominal = self._engine.compute_nominal(
            fid1, fid2, inst.query_type,
            cad_geometry_getter=self._get_cad_geometry,
        )
        deviation = None
        if nominal is not None:
            deviation = value - nominal

        return QueryResult(
            instruction=inst,
            value=value,
            status="ok",
            nominal=nominal,
            deviation=deviation,
        )

    def _resolve_id(self, raw_id: str) -> Optional[str]:
        """Resolve a query ID to a CADFeature.feature_id."""
        # 1. Direct feature_id match
        if self._repo.get(raw_id):
            return raw_id
        # 2. DXF handle match
        fid = self._repo.get_by_handle(raw_id)
        if fid:
            return fid
        # 3. Partial UUID match
        for feat in self._repo.all_features():
            if feat.feature_id.startswith(raw_id):
                return feat.feature_id
        return None

    def _get_cad_geometry(self, feature_id: str) -> Optional[dict]:
        feat = self._repo.get(feature_id)
        return feat.geometry if feat else None

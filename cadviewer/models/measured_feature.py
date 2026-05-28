"""
MeasuredFeature — represents image-fitted geometry from local ROI fitting.

A MeasuredFeature stores the result of fitting actual image edge data
within a CAD-predicted ROI. It is distinct from CADFeature (which stores
nominal design geometry) and should be used for all dimension computations.

CAD features are priors; MeasuredFeatures are the actual measurements.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .feature import FeatureType


@dataclass
class MeasuredFeature:
    """Fitted geometry from image edge data within a CAD-guided ROI."""

    feature_id: str
    cad_feature_id: str
    feature_type: FeatureType
    fitted_geometry: dict  # pixel coords (same format as CAD geometry dict)
    fitted_geometry_world: dict  # world coords (mm), transformed via affine
    edge_points: np.ndarray  # Nx2 pixel coords used for fitting
    roi_bbox: tuple  # (xmin, ymin, xmax, ymax) in pixels
    residual_error: float  # mean fitting residual (pixels)
    confidence: float  # 0.0 to 1.0
    detection_method: str  # e.g. "radial_edge_sampling"

    def is_valid(self) -> bool:
        """Check if measurement is usable."""
        return self.confidence > 0.2 and self.residual_error < 5.0


class MeasuredFeatureStore:
    """Registry of measured features, indexed by CAD feature ID."""

    def __init__(self) -> None:
        self._by_cad_id: Dict[str, MeasuredFeature] = {}
        self._by_id: Dict[str, MeasuredFeature] = {}

    def add(self, mf: MeasuredFeature) -> None:
        self._by_cad_id[mf.cad_feature_id] = mf
        self._by_id[mf.feature_id] = mf

    def get_by_cad_id(self, cad_id: str) -> Optional[MeasuredFeature]:
        return self._by_cad_id.get(cad_id)

    def get(self, feature_id: str) -> Optional[MeasuredFeature]:
        return self._by_id.get(feature_id)

    def all_measured(self) -> List[MeasuredFeature]:
        return list(self._by_id.values())

    def clear(self) -> None:
        self._by_cad_id.clear()
        self._by_id.clear()

    def count(self) -> int:
        return len(self._by_id)

    def has_measurement(self, cad_id: str) -> bool:
        return cad_id in self._by_cad_id

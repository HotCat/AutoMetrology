"""
MeasuredFeature — represents image-fitted geometry from local ROI fitting.

A MeasuredFeature stores the result of fitting actual image edge data
within a CAD-predicted ROI. It is distinct from CADFeature (which stores
nominal design geometry) and should be used for all dimension computations.

DATA CONTRACT:
  - source_type must be one of: IMAGE_EDGE, FITTED, MEASURED
  - The query evaluator must NEVER access CADFeature.geometry for measured values
  - MeasuredFeature.fitted_geometry_world is the ONLY valid source for measurements

CAD features are priors; MeasuredFeatures are the actual measurements.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from .feature import FeatureType


class GeometrySourceType(Enum):
    """Origin of geometry data — used for audit and validation."""
    CAD = "CAD"                     # Nominal design geometry (from DXF)
    REGISTERED_CAD = "REGISTERED_CAD"  # CAD transformed by registration (not measured)
    IMAGE_EDGE = "IMAGE_EDGE"       # Raw edge points from image gradient
    FITTED = "FITTED"               # Geometry fitted from image edge points
    MEASURED = "MEASURED"           # Final measurement result (IMAGE_EDGE or FITTED)


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
    source_type: str = "IMAGE_EDGE"  # GeometrySourceType value for audit

    def is_valid(self) -> bool:
        """Check if measurement is usable."""
        return self.confidence > 0.2 and self.residual_error < 5.0

    def assert_source_is_image(self) -> None:
        """Assert that geometry comes from image, not CAD.
        Raises AssertionError if source_type indicates CAD origin.
        """
        valid_sources = ("IMAGE_EDGE", "FITTED", "MEASURED")
        if self.source_type not in valid_sources:
            raise AssertionError(
                f"Data contract violation: MeasuredFeature.source_type={self.source_type}, "
                f"must be one of {valid_sources}. "
                f"CAD geometry cannot be used for measurement values."
            )


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

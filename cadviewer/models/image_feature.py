"""
ImageFeature — detected geometric feature from a product image.

Mirrors CADFeature geometry dict conventions for direct comparison:
  LINE:    {x1, y1, x2, y2}
  CIRCLE:  {cx, cy, radius}
  CONTOUR: {points: [(x,y),...], closed: bool}
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np


class ImageFeatureType(Enum):
    LINE = auto()
    CIRCLE = auto()
    ARC = auto()
    CONTOUR = auto()
    POINT = auto()


# Map CAD FeatureType to ImageFeatureType for correspondence matching
CAD_TO_IMAGE_TYPE = {
    "LINE": ImageFeatureType.LINE,
    "CIRCLE": ImageFeatureType.CIRCLE,
    "ARC": ImageFeatureType.ARC,
    "POLYLINE": ImageFeatureType.CONTOUR,
}


@dataclass
class ImageFeature:
    """A detected geometric feature in a product image."""
    feature_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    feature_type: ImageFeatureType = ImageFeatureType.LINE
    geometry: dict = field(default_factory=dict)
    pixel_coords: Optional[np.ndarray] = None  # raw detection points (Nx2)
    fitting_residual: float = 0.0
    confidence: float = 1.0
    roi_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x_min, y_min, x_max, y_max
    detection_method: str = ""  # "hough", "lsq_circle", "contour_fit", etc.

    @property
    def display_name(self) -> str:
        return f"{self.feature_type.name} [{self.feature_id[:8]}]"


class ImageFeatureRepository:
    """Registry of detected image features."""

    def __init__(self) -> None:
        self._features: Dict[str, ImageFeature] = {}
        self._by_type: Dict[ImageFeatureType, List[str]] = {}

    def add(self, feature: ImageFeature) -> None:
        self._features[feature.feature_id] = feature
        self._by_type.setdefault(feature.feature_type, []).append(feature.feature_id)

    def get(self, feature_id: str) -> Optional[ImageFeature]:
        return self._features.get(feature_id)

    def features_by_type(self, ftype: ImageFeatureType) -> List[ImageFeature]:
        return [self._features[fid] for fid in self._by_type.get(ftype, [])]

    def all_features(self) -> List[ImageFeature]:
        return list(self._features.values())

    def type_counts(self) -> Dict[ImageFeatureType, int]:
        return {ft: len(ids) for ft, ids in self._by_type.items()}

    def count(self) -> int:
        return len(self._features)

    def clear(self) -> None:
        self._features.clear()
        self._by_type.clear()

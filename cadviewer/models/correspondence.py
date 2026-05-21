"""
FeatureCorrespondence — semantic mapping between CAD and image features.

Establishes the bidirectional mapping:
  CAD Feature ID ↔ Detected Image Feature ID

with confidence, residual error, and the affine transform used.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .image_feature import ImageFeature


@dataclass
class FeatureCorrespondence:
    """A semantic match between a CAD feature and a detected image feature."""
    correspondence_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cad_feature_id: str = ""
    image_feature_id: str = ""
    confidence: float = 0.0   # 0.0 to 1.0
    residual_error: float = 0.0  # geometric distance after alignment (mm)
    transform_used: Optional[np.ndarray] = None  # 3x3 affine at match time
    match_method: str = ""  # "roi_type", "proximity", "manual"

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.7


class CorrespondenceMap:
    """Bidirectional mapping between CAD and image features."""

    def __init__(self) -> None:
        self._cad_to_corr: Dict[str, str] = {}  # cad_id → corr_id
        self._img_to_corr: Dict[str, str] = {}   # img_id → corr_id
        self._correspondences: Dict[str, FeatureCorrespondence] = {}

    def add(self, corr: FeatureCorrespondence) -> None:
        # Remove any existing correspondence for these features
        if corr.cad_feature_id in self._cad_to_corr:
            old_corr_id = self._cad_to_corr[corr.cad_feature_id]
            old_corr = self._correspondences.pop(old_corr_id, None)
            if old_corr:
                self._img_to_corr.pop(old_corr.image_feature_id, None)
        if corr.image_feature_id in self._img_to_corr:
            old_corr_id = self._img_to_corr[corr.image_feature_id]
            old_corr = self._correspondences.pop(old_corr_id, None)
            if old_corr:
                self._cad_to_corr.pop(old_corr.cad_feature_id, None)

        self._correspondences[corr.correspondence_id] = corr
        self._cad_to_corr[corr.cad_feature_id] = corr.correspondence_id
        self._img_to_corr[corr.image_feature_id] = corr.correspondence_id

    def get_for_cad(self, cad_feature_id: str) -> Optional[FeatureCorrespondence]:
        corr_id = self._cad_to_corr.get(cad_feature_id)
        return self._correspondences.get(corr_id) if corr_id else None

    def get_for_image(self, image_feature_id: str) -> Optional[FeatureCorrespondence]:
        corr_id = self._img_to_corr.get(image_feature_id)
        return self._correspondences.get(corr_id) if corr_id else None

    def all_correspondences(self) -> List[FeatureCorrespondence]:
        return list(self._correspondences.values())

    def confident_correspondences(self, threshold: float = 0.7) -> List[FeatureCorrespondence]:
        return [c for c in self._correspondences.values() if c.confidence >= threshold]

    def remove_cad(self, cad_feature_id: str) -> None:
        corr_id = self._cad_to_corr.pop(cad_feature_id, None)
        if corr_id:
            corr = self._correspondences.pop(corr_id, None)
            if corr:
                self._img_to_corr.pop(corr.image_feature_id, None)

    def clear(self) -> None:
        self._cad_to_corr.clear()
        self._img_to_corr.clear()
        self._correspondences.clear()

    def count(self) -> int:
        return len(self._correspondences)

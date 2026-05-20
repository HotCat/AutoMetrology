"""
FeatureRepository — central registry of all CAD features.

Provides:
  - add / remove / lookup by id
  - queries by type, layer, handle
  - iteration for rendering

This is the single source of truth for the geometry model layer.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from .feature import CADFeature, FeatureType


class FeatureRepository:
    def __init__(self) -> None:
        self._features: Dict[str, CADFeature] = {}
        self._by_type: Dict[FeatureType, List[str]] = defaultdict(list)
        self._by_layer: Dict[str, List[str]] = defaultdict(list)
        self._by_handle: Dict[str, str] = {}

    def add(self, feature: CADFeature) -> None:
        fid = feature.feature_id
        self._features[fid] = feature
        self._by_type[feature.feature_type].append(fid)
        self._by_layer[feature.layer].append(fid)
        if feature.dxf_handle:
            self._by_handle[feature.dxf_handle] = fid

    def get(self, feature_id: str) -> Optional[CADFeature]:
        return self._features.get(feature_id)

    def get_by_handle(self, dxf_handle: str) -> Optional[CADFeature]:
        fid = self._by_handle.get(dxf_handle)
        return self._features.get(fid) if fid else None

    def all_features(self) -> List[CADFeature]:
        return list(self._features.values())

    def features_by_type(self, ftype: FeatureType) -> List[CADFeature]:
        return [self._features[fid] for fid in self._by_type.get(ftype, []) if fid in self._features]

    def features_by_layer(self, layer: str) -> List[CADFeature]:
        return [self._features[fid] for fid in self._by_layer.get(layer, []) if fid in self._features]

    def all_layers(self) -> List[str]:
        return list(self._by_layer.keys())

    def type_counts(self) -> Dict[FeatureType, int]:
        return {ft: len(ids) for ft, ids in self._by_type.items()}

    def count(self) -> int:
        return len(self._features)

    def clear(self) -> None:
        self._features.clear()
        self._by_type.clear()
        self._by_layer.clear()
        self._by_handle.clear()

    def remove(self, feature_id: str) -> None:
        feat = self._features.pop(feature_id, None)
        if feat:
            self._by_type[feat.feature_type].remove(feature_id)
            self._by_layer[feat.layer].remove(feature_id)
            if feat.dxf_handle in self._by_handle:
                del self._by_handle[feat.dxf_handle]

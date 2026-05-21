"""
RegistrationGroup — data model for grouping CAD features into registration anchors.

Registration groups are used for:
  - robust CAD-to-image alignment
  - ICP anchor selection
  - geometric registration constraints

Each group maintains a set of CADFeature IDs and computes aggregate geometry
(centroid, bounding box, type statistics) on demand from the FeatureRepository.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from PySide6.QtGui import QColor

from .feature import CADFeature, FeatureType
from .repository import FeatureRepository

# Distinct color palette for group visualization
_GROUP_COLORS = [
    QColor(255, 100, 100, 180),  # red
    QColor(100, 255, 100, 180),  # green
    QColor(100, 100, 255, 180),  # blue
    QColor(255, 200, 50, 180),   # gold
    QColor(200, 100, 255, 180),  # purple
    QColor(100, 255, 255, 180),  # cyan
    QColor(255, 150, 50, 180),   # orange
    QColor(255, 100, 200, 180),  # pink
]


def _compute_group_geometry(
    feature_ids: List[str], repo: FeatureRepository
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float, float, float]]]:
    """Compute centroid and bounding box for a set of features."""
    if not feature_ids:
        return None, None

    all_x: List[float] = []
    all_y: List[float] = []

    for fid in feature_ids:
        feat = repo.get(fid)
        if not feat:
            continue
        g = feat.geometry
        ft = feat.feature_type

        if ft == FeatureType.LINE:
            all_x.extend([g["x1"], g["x2"]])
            all_y.extend([g["y1"], g["y2"]])
        elif ft in (FeatureType.CIRCLE, FeatureType.ARC):
            cx, cy, r = g["cx"], g["cy"], g.get("radius", 0)
            all_x.extend([cx - r, cx + r])
            all_y.extend([cy - r, cy + r])
        elif ft == FeatureType.POLYLINE:
            for pt in g.get("points", []):
                all_x.append(pt[0])
                all_y.append(pt[1])
        elif ft == FeatureType.SPLINE:
            for pt in (g.get("control_points", []) or g.get("fit_points", [])):
                all_x.append(pt[0])
                all_y.append(pt[1])
        elif ft == FeatureType.TEXT:
            h = g.get("height", 2.5)
            all_x.extend([g["x"], g["x"] + h * 3])
            all_y.extend([g["y"], g["y"] + h])

    if not all_x:
        return None, None

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    centroid = ((min_x + max_x) / 2, (min_y + max_y) / 2)
    bbox = (min_x, min_y, max_x, max_y)
    return centroid, bbox


@dataclass
class RegistrationGroup:
    """A named group of CAD features for registration anchoring."""

    group_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "New Group"
    color: QColor = field(default_factory=lambda: QColor(255, 100, 100, 180))
    feature_ids: List[str] = field(default_factory=list)

    def centroid(self, repo: FeatureRepository) -> Optional[Tuple[float, float]]:
        c, _ = _compute_group_geometry(self.feature_ids, repo)
        return c

    def bbox(self, repo: FeatureRepository) -> Optional[Tuple[float, float, float, float]]:
        _, b = _compute_group_geometry(self.feature_ids, repo)
        return b

    def type_statistics(self, repo: FeatureRepository) -> Dict[FeatureType, int]:
        stats: Dict[FeatureType, int] = {}
        for fid in self.feature_ids:
            feat = repo.get(fid)
            if feat:
                stats[feat.feature_type] = stats.get(feat.feature_type, 0) + 1
        return stats

    @property
    def feature_count(self) -> int:
        return len(self.feature_ids)

    def contains(self, feature_id: str) -> bool:
        return feature_id in self.feature_ids


class RegistrationManager:
    """Manages all registration groups with CRUD operations and reverse lookup."""

    def __init__(self, repo: FeatureRepository) -> None:
        self._repo = repo
        self._groups: Dict[str, RegistrationGroup] = {}
        self._feature_to_group: Dict[str, str] = {}
        self._color_index = 0

    def _next_color(self) -> QColor:
        color = _GROUP_COLORS[self._color_index % len(_GROUP_COLORS)]
        self._color_index += 1
        return QColor(color)

    def create_group(self, name: Optional[str] = None) -> RegistrationGroup:
        group = RegistrationGroup(
            name=name or f"Group {len(self._groups) + 1}",
            color=self._next_color(),
        )
        self._groups[group.group_id] = group
        return group

    def delete_group(self, group_id: str) -> None:
        group = self._groups.pop(group_id, None)
        if group:
            for fid in group.feature_ids:
                self._feature_to_group.pop(fid, None)

    def rename_group(self, group_id: str, new_name: str) -> None:
        group = self._groups.get(group_id)
        if group:
            group.name = new_name

    def add_feature_to_group(self, group_id: str, feature_id: str) -> bool:
        group = self._groups.get(group_id)
        if not group:
            return False
        if feature_id in self._feature_to_group:
            return False  # already in a group
        group.feature_ids.append(feature_id)
        self._feature_to_group[feature_id] = group_id
        return True

    def remove_feature_from_group(self, group_id: str, feature_id: str) -> None:
        group = self._groups.get(group_id)
        if group and feature_id in group.feature_ids:
            group.feature_ids.remove(feature_id)
            self._feature_to_group.pop(feature_id, None)

    def move_feature_to_group(self, group_id: str, feature_id: str) -> bool:
        current_gid = self._feature_to_group.get(feature_id)
        if current_gid == group_id:
            return False
        if current_gid:
            self.remove_feature_from_group(current_gid, feature_id)
        return self.add_feature_to_group(group_id, feature_id)

    def get_group(self, group_id: str) -> Optional[RegistrationGroup]:
        return self._groups.get(group_id)

    def get_group_for_feature(self, feature_id: str) -> Optional[RegistrationGroup]:
        gid = self._feature_to_group.get(feature_id)
        return self._groups.get(gid) if gid else None

    def all_groups(self) -> List[RegistrationGroup]:
        return list(self._groups.values())

    def group_count(self) -> int:
        return len(self._groups)

    def clear(self) -> None:
        self._groups.clear()
        self._feature_to_group.clear()
        self._color_index = 0

    def set_repository(self, repo: FeatureRepository) -> None:
        self._repo = repo
        self.clear()

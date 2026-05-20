"""
CADFeature — immutable data object representing a single geometric entity.

Each feature carries:
  - unique id (uuid4)
  - feature type (line, circle, arc, polyline, spline, dimension, text, hatch)
  - geometry data (type-specific params)
  - original DXF entity handle for traceability
  - rendered AIS object reference (set by renderer, None until rendered)
  - layer name
  - measurement metadata placeholder for future machine vision integration
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


class FeatureType(Enum):
    LINE = auto()
    CIRCLE = auto()
    ARC = auto()
    POLYLINE = auto()
    SPLINE = auto()
    DIMENSION = auto()
    TEXT = auto()
    HATCH = auto()
    POINT = auto()


@dataclass
class MeasurementMetadata:
    """Placeholder for future machine vision measurement data."""
    nominal_value: Optional[float] = None
    measured_value: Optional[float] = None
    tolerance_plus: Optional[float] = None
    tolerance_minus: Optional[float] = None
    deviation: Optional[float] = None
    is_passing: Optional[bool] = None
    image_registration: Optional[dict] = None  # affine transform params


@dataclass
class CADFeature:
    feature_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    feature_type: FeatureType = FeatureType.LINE
    geometry: dict = field(default_factory=dict)
    dxf_handle: str = ""
    layer: str = "0"
    color: int = 7  # DXF default white

    # Populated by the rendering layer after AIS display
    ais_object: Any = None

    # Future measurement integration
    measurement: MeasurementMetadata = field(default_factory=MeasurementMetadata)

    @property
    def display_name(self) -> str:
        handle_part = self.dxf_handle[:8] if self.dxf_handle else self.feature_id[:8]
        return f"{self.feature_type.name} [{handle_part}]"

    def geometry_summary(self) -> dict:
        """Return a human-readable summary of the geometry parameters."""
        summary = {"type": self.feature_type.name}
        g = self.geometry
        if self.feature_type == FeatureType.LINE:
            summary["start"] = f"({g.get('x1', 0):.4f}, {g.get('y1', 0):.4f})"
            summary["end"] = f"({g.get('x2', 0):.4f}, {g.get('y2', 0):.4f})"
            dx = g.get('x2', 0) - g.get('x1', 0)
            dy = g.get('y2', 0) - g.get('y1', 0)
            summary["length"] = f"{(dx**2 + dy**2)**0.5:.4f}"
        elif self.feature_type in (FeatureType.CIRCLE, FeatureType.ARC):
            summary["center"] = f"({g.get('cx', 0):.4f}, {g.get('cy', 0):.4f})"
            summary["radius"] = f"{g.get('radius', 0):.4f}"
            if self.feature_type == FeatureType.ARC:
                summary["start_angle"] = f"{g.get('start_angle', 0):.2f} deg"
                summary["end_angle"] = f"{g.get('end_angle', 0):.2f} deg"
        elif self.feature_type == FeatureType.POLYLINE:
            pts = g.get('points', [])
            summary["vertex_count"] = len(pts)
            summary["closed"] = g.get('closed', False)
            if pts:
                summary["extent"] = (
                    f"({pts[0][0]:.2f}, {pts[0][1]:.2f}) -> "
                    f"({pts[-1][0]:.2f}, {pts[-1][1]:.2f})"
                )
        elif self.feature_type == FeatureType.SPLINE:
            summary["degree"] = g.get('degree', 0)
            summary["control_points"] = len(g.get('control_points', []))
            summary["fit_points"] = len(g.get('fit_points', []))
        elif self.feature_type == FeatureType.TEXT:
            summary["text"] = g.get('text', '')
            summary["position"] = f"({g.get('x', 0):.2f}, {g.get('y', 0):.2f})"
            summary["height"] = f"{g.get('height', 0):.2f}"
            if g.get('rotation', 0) != 0:
                summary["rotation"] = f"{g.get('rotation', 0):.1f}°"
        elif self.feature_type == FeatureType.DIMENSION:
            summary["dim_type"] = g.get('dim_type', 0)
            if g.get('text'):
                summary["text"] = g.get('text', '')
        return summary

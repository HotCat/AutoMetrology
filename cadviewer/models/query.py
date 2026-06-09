"""
Query models — data structures for the measurement query language.

Geometry source contract:
  - geometry_source = "MEASURED"  → value from image-fitted MeasuredFeature
  - geometry_source = "NONE"      → no image measurement available
  The query evaluator must NEVER set geometry_source to "CAD".
  CAD geometry is used only for the nominal reference value.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class QueryType(Enum):
    CIRCLE_DISTANCE = auto()  # circles(ID1, ID2) → center distance
    LINE_DISTANCE = auto()    # lines(ID1, ID2) → perpendicular distance
    ARC_RADIUS = auto()       # arcs(ID) → fitted arc radius
    CIRCLE_RADIUS = auto()    # circle(ID) → fitted circle radius


@dataclass
class QueryInstruction:
    raw_text: str = ""
    query_type: QueryType = QueryType.CIRCLE_DISTANCE
    feature_id_1: str = ""
    feature_id_2: str = ""
    line_number: int = 0


@dataclass
class QueryResult:
    instruction: Optional[QueryInstruction] = None
    value: Optional[float] = None  # mm (from image-fitted geometry only)
    unit: str = "mm"
    status: str = "pending"  # "ok", "error", "no_measurement"
    error_message: str = ""
    nominal: Optional[float] = None  # CAD nominal value
    deviation: Optional[float] = None  # measured - nominal
    geometry_source: str = "NONE"  # "MEASURED", "NONE" — never "CAD"
    # Audit trail: which geometry was used for each feature
    feature_geometry_audit: Optional[dict] = None

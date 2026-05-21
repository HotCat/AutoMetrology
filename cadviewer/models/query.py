"""
Query models — data structures for the measurement query language.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class QueryType(Enum):
    CIRCLE_DISTANCE = auto()  # circles(ID1, ID2) → center distance
    LINE_DISTANCE = auto()    # lines(ID1, ID2) → perpendicular distance


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
    value: Optional[float] = None  # mm
    unit: str = "mm"
    status: str = "pending"  # "ok", "error", "no_correspondence"
    error_message: str = ""
    nominal: Optional[float] = None  # CAD nominal value
    deviation: Optional[float] = None  # measured - nominal

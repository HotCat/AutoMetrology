"""
QueryParser — parse line-oriented measurement query files.

Grammar:
  query_file  := query_line*
  query_line  := instruction | comment | blank
  instruction := func_name '(' id ',' id ')'
  func_name   := 'circles' | 'lines'
  id          := [A-Za-z0-9_]+
  comment     := '#' ...
"""

from __future__ import annotations

import re
from typing import List

from ..models.query import QueryInstruction, QueryType

# Regex for parsing a single instruction line
_INSTRUCTION_RE = re.compile(
    r'^\s*(circles|lines)\s*\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)\s*$'
)


class QueryParser:
    """Parse measurement query text into QueryInstruction list."""

    def parse(self, text: str) -> List[QueryInstruction]:
        instructions = []
        for line_no, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            inst = self._parse_line(line, line_no)
            if inst:
                instructions.append(inst)
            else:
                raise ValueError(
                    f"Syntax error at line {line_no}: '{line}'\n"
                    f"Expected: circles(ID1, ID2) or lines(ID1, ID2)"
                )
        return instructions

    def parse_file(self, path: str) -> List[QueryInstruction]:
        with open(path, 'r') as f:
            return self.parse(f.read())

    def _parse_line(self, line: str, line_no: int) -> QueryInstruction | None:
        m = _INSTRUCTION_RE.match(line)
        if not m:
            return None
        func_name, id1, id2 = m.group(1), m.group(2), m.group(3)
        query_type = (
            QueryType.CIRCLE_DISTANCE if func_name == "circles"
            else QueryType.LINE_DISTANCE
        )
        return QueryInstruction(
            raw_text=line,
            query_type=query_type,
            feature_id_1=id1,
            feature_id_2=id2,
            line_number=line_no,
        )

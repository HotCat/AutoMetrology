"""Tests for measurement query parsing."""

import unittest

from cadviewer.measurement.query_parser import QueryParser
from cadviewer.models.query import QueryType


class QueryParserTest(unittest.TestCase):
    def test_parser_accepts_exploded_polyline_segment_handles(self) -> None:
        instructions = QueryParser().parse(
            "lines(AB8E:7, AB8E:3), 0.1\n"
            "circle(A1-B2:C3), 0.05\n"
        )

        self.assertEqual(len(instructions), 2)
        self.assertEqual(instructions[0].query_type, QueryType.LINE_DISTANCE)
        self.assertEqual(instructions[0].feature_id_1, "AB8E:7")
        self.assertEqual(instructions[0].feature_id_2, "AB8E:3")
        self.assertEqual(instructions[0].tolerance_abs, 0.1)
        self.assertEqual(instructions[1].query_type, QueryType.CIRCLE_RADIUS)
        self.assertEqual(instructions[1].feature_id_1, "A1-B2:C3")
        self.assertEqual(instructions[1].tolerance_abs, 0.05)


if __name__ == "__main__":
    unittest.main()

"""
ResultWriter — export measurement results to text files.

Geometry source is reported in output:
  "MEASURED" → value from image-fitted geometry (valid measurement)
  "NONE"     → no image measurement available
"""

from __future__ import annotations

from typing import List

from ..models.query import QueryResult


class ResultWriter:
    """Write query results to plain text or CSV."""

    @staticmethod
    def write_results(results: List[QueryResult], path: str) -> None:
        """Write results as plain text, one line per query."""
        with open(path, 'w') as f:
            for r in results:
                f.write(ResultWriter.format_result(r) + "\n")

    @staticmethod
    def format_result(r: QueryResult) -> str:
        source_tag = f" [source: {r.geometry_source}]" if r.geometry_source else ""

        if r.status == "ok" and r.instruction:
            line = f"{r.instruction.raw_text} = {r.value:.3f} {r.unit}"
            if r.nominal is not None:
                line += f" [nominal: {r.nominal:.3f}]"
            if r.deviation is not None:
                sign = "+" if r.deviation >= 0 else ""
                line += f" [dev: {sign}{r.deviation:.3f}]"
            line += source_tag
            return line
        elif r.status == "no_measurement" and r.instruction:
            return (
                f"{r.instruction.raw_text} = NO MEASUREMENT: "
                f"{r.error_message}"
                f"{source_tag}"
            )
        elif r.instruction:
            return f"{r.instruction.raw_text} = ERROR: {r.error_message}"
        else:
            return f"ERROR: {r.error_message}"

    @staticmethod
    def write_csv(results: List[QueryResult], path: str) -> None:
        """Write results as CSV for SPC integration."""
        with open(path, 'w') as f:
            f.write("query,type,id1,id2,status,value_mm,nominal_mm,deviation_mm,geometry_source,error\n")
            for r in results:
                inst = r.instruction
                if inst:
                    row = [
                        inst.raw_text,
                        inst.query_type.name,
                        inst.feature_id_1,
                        inst.feature_id_2,
                        r.status,
                        f"{r.value:.4f}" if r.value is not None else "",
                        f"{r.nominal:.4f}" if r.nominal is not None else "",
                        f"{r.deviation:.4f}" if r.deviation is not None else "",
                        r.geometry_source,
                        r.error_message,
                    ]
                else:
                    row = ["", "", "", "", r.status, "", "", "", r.geometry_source, r.error_message]
                f.write(",".join(row) + "\n")

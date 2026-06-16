#!/usr/bin/env python3
"""Convert closed rectangular DXF polylines into LINE entities."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ezdxf


def _polyline_points(entity) -> list[tuple[float, float]]:
    if entity.dxftype() == "LWPOLYLINE":
        if not entity.closed:
            return []
        return [(float(x), float(y)) for x, y in entity.get_points(format="xy")]
    if entity.dxftype() == "POLYLINE":
        flags = entity.dxf.flags if hasattr(entity.dxf, "flags") else 0
        if not bool(flags & 1):
            return []
        return [
            (float(vertex.dxf.location.x), float(vertex.dxf.location.y))
            for vertex in entity.vertices
        ]
    return []


def _entity_handle(entity) -> str:
    return str(getattr(entity.dxf, "handle", "") or "")


def _copy_common_dxf_attrs(src, dst) -> None:
    for attr in ("layer", "color", "linetype", "lineweight"):
        if src.dxf.hasattr(attr):
            try:
                setattr(dst.dxf, attr, getattr(src.dxf, attr))
            except Exception:
                pass


def explode_rectangular_polylines(
    input_path: str | Path,
    output_path: str | Path,
    handles: set[str] | None = None,
    keep_source: bool = False,
) -> int:
    doc = ezdxf.readfile(str(input_path))
    msp = doc.modelspace()
    handle_filter = {h.lower() for h in handles or set() if h}
    exploded = 0
    to_delete = []

    for entity in list(msp):
        if entity.dxftype() not in {"POLYLINE", "LWPOLYLINE"}:
            continue
        handle = _entity_handle(entity)
        if handle_filter and handle.lower() not in handle_filter:
            continue
        points = _polyline_points(entity)
        if len(points) != 4:
            continue
        for idx, start in enumerate(points):
            end = points[(idx + 1) % len(points)]
            line = msp.add_line(start, end)
            _copy_common_dxf_attrs(entity, line)
        exploded += 1
        if not keep_source:
            to_delete.append(entity)

    for entity in to_delete:
        msp.delete_entity(entity)

    doc.saveas(str(output_path))
    return exploded


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument(
        "--handle",
        action="append",
        default=[],
        help="Only explode this polyline handle. Can be passed more than once.",
    )
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="Keep source polylines instead of replacing them.",
    )
    args = parser.parse_args(argv)

    count = explode_rectangular_polylines(
        args.input,
        args.output,
        handles=set(args.handle),
        keep_source=bool(args.keep_source),
    )
    print(f"exploded_rectangular_polylines={count}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

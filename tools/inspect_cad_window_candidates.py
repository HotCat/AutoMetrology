#!/usr/bin/env python3
"""Print CAD entities relevant to generic window registration selection."""

from __future__ import annotations

import argparse
import math
import sys

from cadviewer.models.feature import FeatureType
from cadviewer.parsers.dxf_importer import DXFImporter


def _line_length_angle(geom: dict) -> tuple[float, float]:
    dx = float(geom["x2"]) - float(geom["x1"])
    dy = float(geom["y2"]) - float(geom["y1"])
    return math.hypot(dx, dy), abs(math.degrees(math.atan2(dy, dx))) % 180.0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dxf")
    parser.add_argument("--prefix", action="append", default=[])
    parser.add_argument("--top-lines", type=int, default=80)
    args = parser.parse_args(argv)

    repo = DXFImporter().import_file(args.dxf)
    print("feature_count", repo.count())
    print("type_counts", {key.name: value for key, value in repo.type_counts().items()})

    for prefix in args.prefix:
        print("PREFIX", prefix)
        needle = prefix.lower()
        for feature in repo.all_features():
            handle = (feature.dxf_handle or "").lower()
            if (
                feature.feature_id.lower().startswith(needle)
                or handle.startswith(needle)
            ):
                print(
                    feature.display_name,
                    "id", feature.feature_id,
                    "handle", feature.dxf_handle,
                    "layer", feature.layer,
                    "type", feature.feature_type.name,
                    "geom", feature.geometry,
                )

    print("POLYLINES")
    for feature in repo.features_by_type(FeatureType.POLYLINE):
        points = feature.geometry.get("points", [])
        if not points:
            continue
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        print(
            feature.display_name,
            "id", feature.feature_id,
            "handle", feature.dxf_handle,
            "layer", feature.layer,
            "n", len(points),
            "closed", feature.geometry.get("closed"),
            "bbox", (min(xs), min(ys), max(xs), max(ys)),
        )

    print("LONG_LINES")
    rows = []
    for feature in repo.features_by_type(FeatureType.LINE):
        length, angle = _line_length_angle(feature.geometry)
        rows.append((
            length,
            angle,
            feature.display_name,
            feature.feature_id,
            feature.dxf_handle,
            feature.layer,
            feature.geometry,
        ))
    for row in sorted(rows, reverse=True)[:args.top_lines]:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

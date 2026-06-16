#!/usr/bin/env python3
"""Probe window-line registration behavior on a CAD/image pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from cadviewer.parsers.dxf_importer import DXFImporter
from cadviewer.registration import affine_solver
from cadviewer.registration.window_line_registration import register_window_lines

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"cv2 is required: {exc}") from exc


PROVIDED_EDGES = ["30b36028", "7033b964", "8d337da5", "6a358eec"]


def _draw_overlay(path: str, image: np.ndarray, result) -> None:
    panel = image.copy()
    inv = np.linalg.inv(result.transform)
    projected = affine_solver.apply_projective(inv, result.cad_corners)
    cv2.polylines(
        panel,
        [np.round(projected).astype(np.int32)],
        True,
        (255, 0, 0),
        5,
        lineType=cv2.LINE_AA,
    )
    cv2.polylines(
        panel,
        [np.round(result.image_corners).astype(np.int32)],
        True,
        (0, 255, 0),
        3,
        lineType=cv2.LINE_AA,
    )
    cv2.imwrite(path, panel)


def _summarize_result(result) -> dict:
    return {
        "line_handles": result.line_handles,
        "component_bbox": list(result.component_bbox),
        "confidence": result.confidence,
        "transform_model": result.transform_model,
        "homography_safety": result.homography_safety,
        "side_positions": result.side_positions,
        "image_corners": result.image_corners.tolist(),
        "cad_corners": result.cad_corners.tolist(),
        "transform": result.transform.tolist(),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dxf", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--overlay", default="/tmp/window_register_new_cad.png")
    parser.add_argument("--pixel-size-mm", type=float, default=0.01)
    parser.add_argument("--edge", action="append", default=[])
    args = parser.parse_args(argv)

    repo = DXFImporter().import_file(args.dxf)
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot load image: {args.image}")

    output = {
        "dxf": args.dxf,
        "image": args.image,
        "feature_count": repo.count(),
        "provided_edges": {},
        "runs": {},
    }
    provided_edges = args.edge or PROVIDED_EDGES
    for token in provided_edges:
        feature = repo.get_by_handle(token)
        if feature is None:
            for candidate in repo.all_features():
                if candidate.feature_id.lower().startswith(token.lower()):
                    feature = candidate
                    break
        output["provided_edges"][token] = {
            "display_name": feature.display_name if feature else None,
            "geometry": feature.geometry if feature else None,
        }

    for label, edge_tokens in (
        ("default", None),
        ("provided_edges", provided_edges),
    ):
        try:
            result = register_window_lines(
                repo,
                image,
                edge_tokens=edge_tokens,
                pixel_size_mm=args.pixel_size_mm,
                prefer_homography=True,
            )
            output["runs"][label] = {
                "ok": True,
                "result": _summarize_result(result),
            }
            if label == "provided_edges":
                _draw_overlay(args.overlay, image, result)
                output["overlay"] = args.overlay
        except Exception as exc:
            output["runs"][label] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

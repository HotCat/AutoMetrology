#!/usr/bin/env python3
"""Reproduce Hongyi dark-window registration anisotropy on a saved capture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from cadviewer.calibration.residual_map import residual_map_from_config
from cadviewer.core.config import AppConfig
from cadviewer.measurement.evaluator import QueryEvaluator
from cadviewer.measurement.measurement_pipeline import MeasurementPipeline
from cadviewer.parsers.dxf_importer import DXFImporter
from cadviewer.registration import affine_solver
from cadviewer.registration.auto_correspondence import undistort_if_calibrated
from cadviewer.registration.window_line_registration import register_window_lines

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"cv2 is required: {exc}") from exc


DEFAULT_EDGES = ["AB8E:7", "AB8E:1", "AB8E:3", "AB8E:5"]


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _transform_metrics(matrix: np.ndarray) -> dict:
    m = np.asarray(matrix, dtype=np.float64)
    linear = m[:2, :2]
    _u, singular, _vt = np.linalg.svd(linear)
    sx = float(np.linalg.norm(linear[:, 0]))
    sy = float(np.linalg.norm(linear[:, 1]))
    dot = float(np.dot(linear[:, 0], linear[:, 1]))
    denom = max(sx * sy, 1e-12)
    angle = float(np.degrees(np.arccos(np.clip(dot / denom, -1.0, 1.0))))
    return {
        "column_scale_x": sx,
        "column_scale_y": sy,
        "scale_y_over_x": sy / max(sx, 1e-12),
        "sv_max": float(max(singular)),
        "sv_min": float(min(singular)),
        "sv_anisotropy_pct": float((max(singular) / max(min(singular), 1e-12) - 1.0) * 100.0),
        "axis_angle_deg": angle,
        "det": float(np.linalg.det(linear)),
    }


def _evaluate(repo, image: np.ndarray, query: str, transform: np.ndarray, pixel_size: float, residual_map) -> list[dict]:
    pipeline = MeasurementPipeline(
        repo,
        _to_gray(image),
        transform,
        pixel_size_mm=pixel_size,
        residual_map=residual_map,
        pixel_to_world_transform=transform,
    )
    rows = []
    for result in QueryEvaluator(repo, pipeline).evaluate(query):
        inst = result.instruction
        rows.append({
            "query": inst.raw_text if inst else "",
            "status": result.status,
            "value": result.value,
            "nominal": result.nominal,
            "deviation": result.deviation,
            "geometry_source": result.geometry_source,
            "audit": result.feature_geometry_audit,
        })
    return rows


def _draw_overlay(path: Path, image: np.ndarray, runs: dict) -> None:
    panels = []
    for label, data in runs.items():
        result = data.get("registration")
        transform = np.asarray(data.get("transform"), dtype=np.float64)
        if not result:
            continue
        panel = image.copy()
        cad_corners = np.asarray(result["cad_corners"], dtype=np.float64)
        image_corners = np.asarray(result["image_corners"], dtype=np.float64)
        projected = affine_solver.apply_projective(np.linalg.inv(transform), cad_corners)
        cv2.polylines(panel, [np.round(projected).astype(np.int32)], True, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.polylines(panel, [np.round(image_corners).astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(panel, label, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 7, cv2.LINE_AA)
        cv2.putText(panel, label, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3, cv2.LINE_AA)
        h, w = panel.shape[:2]
        scale = 760.0 / float(w)
        panels.append(cv2.resize(panel, (760, int(round(h * scale))), interpolation=cv2.INTER_AREA))
    if panels:
        cv2.imwrite(str(path), np.vstack(panels))


def _registration_summary(result) -> dict:
    return {
        "line_handles": result.line_handles,
        "method": result.method,
        "transform_model": result.transform_model,
        "homography_safety": result.homography_safety,
        "side_positions": result.side_positions,
        "side_lines": {k: [float(v) for v in vals] for k, vals in result.side_lines.items()},
        "component_bbox": list(result.component_bbox),
        "image_corners": result.image_corners.tolist(),
        "cad_corners": result.cad_corners.tolist(),
    }


def _run_one(repo, image: np.ndarray, query: str, pixel_size: float, edges: list[str], residual_map) -> dict:
    result = register_window_lines(
        repo,
        image,
        edge_tokens=edges,
        pixel_size_mm=pixel_size,
        prefer_homography=True,
        detection_mode="dark",
    )
    transforms = {
        "edge_affine": result.affine,
        "edge_homography": result.homography if result.homography is not None else result.affine,
    }
    output = {
        "registration": _registration_summary(result),
        "transforms": {},
    }
    for name, transform in transforms.items():
        output["transforms"][name] = {
            "transform": np.asarray(transform, dtype=float).tolist(),
            "metrics": _transform_metrics(transform),
            "measurements": _evaluate(repo, image, query, transform, pixel_size, residual_map),
        }
    return output


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dxf", required=True)
    parser.add_argument("--image", default="/tmp/cadrefs_camera_capture.png")
    parser.add_argument("--query", default="query2.txt")
    parser.add_argument("--edge", action="append", default=[])
    parser.add_argument("--json", default="/tmp/hongyi_anisotropy.json")
    parser.add_argument("--overlay", default="/tmp/hongyi_anisotropy_overlay.png")
    args = parser.parse_args(argv)

    cfg = AppConfig.load()
    repo = DXFImporter().import_file(args.dxf)
    raw = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if raw is None:
        raise RuntimeError(f"Cannot load image: {args.image}")
    corrected, applied = undistort_if_calibrated(raw, cfg)
    query = Path(args.query).read_text(encoding="utf-8")
    pixel_size = float(cfg.pixel_size_mm)
    residual_map = residual_map_from_config(cfg)
    edges = args.edge or DEFAULT_EDGES

    output = {
        "dxf": args.dxf,
        "image": args.image,
        "query": args.query,
        "pixel_size_mm": pixel_size,
        "lens": {
            "calibrated": bool(cfg.lens_calibration.calibrated),
            "model": cfg.lens_calibration.calibration_model,
            "rms": cfg.lens_calibration.reprojection_error,
            "image_size": cfg.lens_calibration.image_size,
            "undistort_applied": bool(applied),
        },
        "edges": edges,
        "runs": {
            "raw_dark": _run_one(repo, raw, query, pixel_size, edges, residual_map=None),
            "undistorted_dark": _run_one(repo, corrected, query, pixel_size, edges, residual_map),
            "undistorted_dark_no_residual_map": _run_one(
                repo, corrected, query, pixel_size, edges, residual_map=None,
            ),
        },
    }
    Path(args.json).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    compact = {
        key: {
            name: {
                "metrics": val["metrics"],
                "measurements": [
                    {k: row[k] for k in ("query", "status", "value", "nominal", "deviation")}
                    for row in val["measurements"]
                ],
            }
            for name, val in run["transforms"].items()
        }
        for key, run in output["runs"].items()
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))
    _draw_overlay(Path(args.overlay), corrected, {
        "undistorted_affine": {
            "registration": output["runs"]["undistorted_dark"]["registration"],
            "transform": output["runs"]["undistorted_dark"]["transforms"]["edge_affine"]["transform"],
        },
        "undistorted_homography": {
            "registration": output["runs"]["undistorted_dark"]["registration"],
            "transform": output["runs"]["undistorted_dark"]["transforms"]["edge_homography"]["transform"],
        },
    })
    print(f"wrote {args.json}")
    print(f"wrote {args.overlay}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

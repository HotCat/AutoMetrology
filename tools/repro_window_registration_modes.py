#!/usr/bin/env python3
"""Compare fiducial and window-based registration modes on one saved image."""

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
from cadviewer.registration.window_line_registration import (
    _build_affine_from_sides,
    register_window_lines,
)
from tools.repro_line_jitter import _auto_affine, _profile

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"cv2 is required: {exc}") from exc


GROUND_TRUTH = {
    "lines(9e40c968, 7e6e8eb2), 0.8140": 81.42,
    "lines(9c0bd3a0, 71490463), 0.5565": 55.68,
    "lines(7bedd422, 9c0bd3a0), 0.0717": 7.15,
    "lines(756f8ada, 9e40c968), 0.0718": 7.08,
    "lines(7e6e8eb2, 4595cf5f), 0.1918": 19.12,
    "lines(71490463, a8b8900b), 0.0718": 7.09,
}


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _evaluate(repo, gray, query, transform, pixel_size, residual_map) -> list[dict]:
    pipeline = MeasurementPipeline(
        repo,
        gray,
        transform,
        pixel_size_mm=pixel_size,
        residual_map=residual_map,
    )
    rows = []
    for result in QueryEvaluator(repo, pipeline).evaluate(query):
        inst = result.instruction
        raw = inst.raw_text if inst else ""
        gt = GROUND_TRUTH.get(raw)
        row = {
            "query": raw,
            "status": result.status,
            "value": result.value,
            "nominal": result.nominal,
            "deviation": result.deviation,
            "ground_truth": gt,
            "ground_truth_error": (
                None if gt is None or result.value is None else float(result.value) - gt
            ),
        }
        rows.append(row)
    return rows


def _draw_overlay_panel(
    image: np.ndarray,
    transform: np.ndarray,
    cad_corners: np.ndarray,
    detected_corners: np.ndarray,
    label: str,
) -> np.ndarray:
    panel = image.copy()
    inv = np.linalg.inv(transform)
    projected = affine_solver.apply_projective(inv, cad_corners)
    cv2.polylines(
        panel,
        [np.round(projected).astype(np.int32)],
        True,
        (0, 0, 255),
        5,
        lineType=cv2.LINE_AA,
    )
    cv2.polylines(
        panel,
        [np.round(detected_corners).astype(np.int32)],
        True,
        (0, 255, 0),
        3,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        label,
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (255, 255, 255),
        7,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        label,
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    return panel


def _save_overlay(path: str, image: np.ndarray, modes: dict, result) -> None:
    panels = []
    for label, transform in modes.items():
        panels.append(
            _draw_overlay_panel(
                image,
                transform,
                result.cad_corners,
                result.image_corners,
                label,
            )
        )
    h, w = panels[0].shape[:2]
    scale = 640.0 / float(w)
    resized = [
        cv2.resize(p, (640, int(round(h * scale))), interpolation=cv2.INTER_AREA)
        for p in panels
    ]
    top = np.hstack(resized[:2])
    bottom = np.hstack(resized[2:])
    canvas = np.vstack([top, bottom])
    cv2.imwrite(path, canvas)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dxf", default="xintai.dxf")
    parser.add_argument("--query", default="query2.txt")
    parser.add_argument("--image", default="/tmp/cadrefs_camera_capture.png")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--overlay", default="/tmp/cadrefs_window_registration_modes_overlay.png")
    parser.add_argument("--save-json", default="")
    args = parser.parse_args(argv)

    cfg = AppConfig.load()
    profile = _profile(cfg, args.profile)
    repo = DXFImporter().import_file(args.dxf)
    query = Path(args.query).read_text(encoding="utf-8")
    frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Cannot load image: {args.image}")

    image, undistorted = undistort_if_calibrated(frame, cfg)
    gray = _to_gray(image)
    residual_map = residual_map_from_config(cfg)
    pixel_size = float(profile.get("pixel_size_mm") or cfg.pixel_size_mm)

    two_fiducial, fiducial_reg = _auto_affine(image, cfg, profile, repo)
    window_result = register_window_lines(
        repo,
        image,
        pixel_size_mm=pixel_size,
        prefer_homography=True,
    )
    window_axis_affine = _build_affine_from_sides(
        repo,
        window_result.line_handles,
        window_result.side_positions,
    )

    modes = {
        "two_fiducial": two_fiducial,
        "window_axis_affine": window_axis_affine,
        "edge_affine": window_result.affine,
    }
    if window_result.homography is not None:
        modes["edge_homography"] = window_result.homography
    else:
        modes["edge_homography"] = window_result.affine

    output = {
        "image": args.image,
        "undistorted": bool(undistorted),
        "pixel_size_mm": pixel_size,
        "selected_window_model": window_result.transform_model,
        "homography_safety": window_result.homography_safety,
        "fiducial_registration": fiducial_reg,
        "window": {
            "side_positions": window_result.side_positions,
            "component_bbox": list(window_result.component_bbox),
            "confidence": window_result.confidence,
            "image_corners": window_result.image_corners.tolist(),
            "cad_corners": window_result.cad_corners.tolist(),
        },
        "modes": {},
    }
    for label, transform in modes.items():
        output["modes"][label] = {
            "transform": np.asarray(transform, dtype=float).tolist(),
            "results": _evaluate(repo, gray, query, transform, pixel_size, residual_map),
        }

    _save_overlay(args.overlay, image, modes, window_result)
    output["overlay"] = args.overlay
    print(json.dumps(output, indent=2, ensure_ascii=False))
    if args.save_json:
        Path(args.save_json).write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""Compare current two-point pixel scaling with saved homography correction."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from cadviewer.calibration.coordinate_correction import CoordinateTransformer
from cadviewer.core.config import AppConfig
from cadviewer.measurement.evaluator import QueryEvaluator
from cadviewer.measurement.measurement_pipeline import MeasurementPipeline
from cadviewer.measurement.query_parser import QueryParser
from cadviewer.models.query import QueryType
from cadviewer.parsers.dxf_importer import DXFImporter
from cadviewer.registration import affine_solver
from cadviewer.registration.auto_correspondence import (
    detect_circle_in_roi,
    undistort_if_calibrated,
)
from cadviewer.registration.strategy import TeachICPStrategy

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


def _profile(cfg: AppConfig, name: str | None) -> dict:
    selected = name or cfg.active_production_profile
    for profile in cfg.production_profiles:
        if isinstance(profile, dict) and profile.get("name") == selected:
            return profile
    raise SystemExit(f"Production profile not found: {selected!r}")


def _camera_frame(cfg: AppConfig, profile: dict) -> np.ndarray:
    from cadviewer.camera.device import CameraSettings, MindVisionCamera
    from cadviewer.camera.driver import mvsdk

    camera = MindVisionCamera()
    devices = camera.enumerate_devices()
    if not devices:
        raise RuntimeError("No MindVision camera detected")
    camera.open(devices[0]["dev_info"])
    try:
        cam_cfg = profile.get("camera") or {}
        settings = CameraSettings(
            exposure_us=int(cam_cfg.get("exposure_us", cfg.camera.exposure_us)),
            gamma=int(cam_cfg.get("gamma", cfg.camera.gamma)),
            contrast=int(cam_cfg.get("contrast", cfg.camera.contrast)),
            analog_gain=int(cam_cfg.get("analog_gain", cfg.camera.analog_gain)),
            ae_enabled=bool(cam_cfg.get("ae_enabled", cfg.camera.ae_enabled)),
            reverse_x=bool(cam_cfg.get("reverse_x", cfg.camera.reverse_x)),
            reverse_y=bool(cam_cfg.get("reverse_y", cfg.camera.reverse_y)),
        )
        camera.apply_settings(settings)
        camera.set_trigger_mode()
        time.sleep(0.05)
        mvsdk.CameraSoftTrigger(camera._hCamera)
        frame = camera._grab_frame(timeout_ms=2000)
        if frame is None:
            raise RuntimeError("Camera trigger returned no frame")
        return frame
    finally:
        camera.close()


def _auto_registration(image: np.ndarray, cfg: AppConfig, profile: dict, repo):
    auto = profile.get("auto_correspondence") or {}
    cad_fids = auto.get("cad_fiducials") or []
    rois = auto.get("image_rois") or []
    if len(cad_fids) < 2 or len(rois) < 2:
        raise RuntimeError("Profile is missing two auto-correspondence fiducials/ROIs")

    pixel_size = float(profile.get("pixel_size_mm") or cfg.pixel_size_mm)
    detections = []
    for idx in range(2):
        cad = cad_fids[idx]
        feat = repo.get(cad.get("feature_id", "")) or repo.get_by_handle(cad.get("dxf_handle", ""))
        if feat is None:
            raise RuntimeError(f"Cannot resolve fiducial {idx + 1}: {cad}")
        expected_r = float(feat.geometry.get("radius", 0.0)) / max(pixel_size, 1e-9)
        det = detect_circle_in_roi(image, tuple(rois[idx]), expected_radius_px=expected_r)
        if det is None:
            raise RuntimeError(f"No fiducial detected in ROI {idx + 1}: {rois[idx]}")
        detections.append(det)

    cad_points = [
        {"world": list(map(float, cad_fids[0]["world"]))},
        {"world": list(map(float, cad_fids[1]["world"]))},
    ]
    image_points = [
        {"pixel": [float(detections[0].center[0]), float(detections[0].center[1])]},
        {"pixel": [float(detections[1].center[0]), float(detections[1].center[1])]},
    ]
    registration = TeachICPStrategy._compute_transform_from_points(
        cad_points, image_points, pixel_size,
    )
    affine = TeachICPStrategy._compute_pixel_to_world_transform(registration, pixel_size)
    return affine, cad_points, image_points, detections, pixel_size


def _homography_pixel_to_world(cfg: AppConfig, cad_points: list[dict], image_points: list[dict]):
    lc = cfg.lens_calibration
    if not lc.coordinate_correction or lc.correction_model_type not in {"homography", "affine"}:
        return None

    transformer = CoordinateTransformer()
    if not transformer.load_model(lc.coordinate_correction, lc.correction_model_type):
        return None

    image_px = np.array([p["pixel"] for p in image_points], dtype=np.float64)
    image_metric = transformer.transform(image_px)
    cad_metric = np.array([p["world"] for p in cad_points], dtype=np.float64)
    plane_to_world = affine_solver.solve_similarity(image_metric, cad_metric)

    if lc.correction_model_type == "homography":
        h = np.asarray(lc.coordinate_correction["homography"], dtype=np.float64)
        return plane_to_world @ h
    if lc.correction_model_type == "affine":
        a = np.asarray(lc.coordinate_correction["affine"], dtype=np.float64)
        h = np.vstack([a, np.array([0.0, 0.0, 1.0])])
        return plane_to_world @ h
    return None


def _projective_apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    hom = np.column_stack([pts, np.ones(len(pts))])
    mapped = (matrix @ hom.T).T
    return mapped[:, :2] / mapped[:, 2:3]


def _line_distance(g1: dict, g2: dict) -> float:
    x1, y1 = g1["x1"], g1["y1"]
    x2, y2 = g1["x2"], g1["y2"]
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-12:
        return 0.0
    nx, ny = -dy / length, dx / length
    d1 = abs((g2["x1"] - x1) * nx + (g2["y1"] - y1) * ny)
    d2 = abs((g2["x2"] - x1) * nx + (g2["y2"] - y1) * ny)
    return (d1 + d2) / 2.0


def _recompute_rows(repo, pipeline: MeasurementPipeline, query: str, pixel_to_world: np.ndarray):
    rows = []
    parser = QueryParser()
    evaluator = QueryEvaluator(repo, pipeline)
    # Run existing evaluator first so pair-aware line fitting fills audit data.
    current_results = evaluator.evaluate(query)
    instructions = parser.parse(query)
    for result, inst in zip(current_results, instructions):
        row = {
            "query": inst.raw_text,
            "current": result.value,
            "nominal": result.nominal,
            "homography": None,
        }
        if inst.query_type == QueryType.LINE_DISTANCE and result.feature_geometry_audit:
            audit = result.feature_geometry_audit
            f1 = audit.get("feature_1", {}).get("fitted_geometry_px")
            f2 = audit.get("feature_2", {}).get("fitted_geometry_px")
            if f1 and f2:
                pts = np.array([
                    [f1["x1"], f1["y1"]], [f1["x2"], f1["y2"]],
                    [f2["x1"], f2["y1"]], [f2["x2"], f2["y2"]],
                ], dtype=np.float64)
                world = _projective_apply(pixel_to_world, pts)
                g1 = {"x1": world[0, 0], "y1": world[0, 1], "x2": world[1, 0], "y2": world[1, 1]}
                g2 = {"x1": world[2, 0], "y1": world[2, 1], "x2": world[3, 0], "y2": world[3, 1]}
                row["homography"] = round(_line_distance(g1, g2), 4)
        gt = GROUND_TRUTH.get(inst.raw_text)
        if gt is not None:
            row["ground_truth"] = gt
            if row["current"] is not None:
                row["current_error"] = round(float(row["current"]) - gt, 4)
            if row["homography"] is not None:
                row["homography_error"] = round(float(row["homography"]) - gt, 4)
        rows.append(row)
    return rows


def _print_summary(samples: list[list[dict]], key: str) -> None:
    print(f"\n{key} summary")
    for query in [row["query"] for row in samples[0]]:
        vals = [row[key] for sample in samples for row in sample if row["query"] == query and row[key] is not None]
        if not vals:
            continue
        gt = GROUND_TRUTH.get(query)
        err = statistics.mean(vals) - gt if gt is not None else float("nan")
        print(
            f"{query:42s} n={len(vals):2d} mean={statistics.mean(vals):.5f} "
            f"range={max(vals) - min(vals):.5f} std={statistics.pstdev(vals):.5f} "
            f"err={err:+.5f}"
        )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dxf", default="xintai.dxf")
    parser.add_argument("--query", default="query2.txt")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--save-json", default="")
    args = parser.parse_args(argv)

    cfg = AppConfig.load()
    profile = _profile(cfg, args.profile)
    repo = DXFImporter().import_file(args.dxf)
    query = Path(args.query).read_text(encoding="utf-8")

    samples = []
    for idx in range(args.frames):
        frame = _camera_frame(cfg, profile)
        image, undistorted = undistort_if_calibrated(frame, cfg)
        affine, cad_points, image_points, detections, pixel_size = _auto_registration(
            image, cfg, profile, repo,
        )
        pixel_to_world = _homography_pixel_to_world(cfg, cad_points, image_points)
        if pixel_to_world is None:
            raise RuntimeError("No calibrated coordinate correction is available")

        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] == 1:
            gray = image[:, :, 0]
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pipeline = MeasurementPipeline(repo, gray, affine, pixel_size_mm=pixel_size)
        rows = _recompute_rows(repo, pipeline, query, pixel_to_world)
        samples.append(rows)
        compact = ", ".join(
            f"{row['current']:.4f}->{row['homography']:.4f}"
            if row["current"] is not None and row["homography"] is not None
            else "no_measurement"
            for row in rows
        )
        print(f"frame {idx + 1:02d} undistorted={undistorted}: {compact}")

    _print_summary(samples, "current")
    _print_summary(samples, "homography")
    if args.save_json:
        Path(args.save_json).write_text(
            json.dumps(samples, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

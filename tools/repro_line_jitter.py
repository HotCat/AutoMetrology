#!/usr/bin/env python3
"""Reproduce production line-fit jitter with the active camera/profile."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from cadviewer.core.config import AppConfig
from cadviewer.measurement.evaluator import QueryEvaluator
from cadviewer.measurement.measurement_pipeline import MeasurementPipeline
from cadviewer.parsers.dxf_importer import DXFImporter
from cadviewer.registration.auto_correspondence import (
    detect_circle_in_roi,
    undistort_if_calibrated,
)
from cadviewer.registration.strategy import TeachICPStrategy

try:
    import cv2
except ImportError as exc:  # pragma: no cover - diagnostic script
    raise SystemExit(f"cv2 is required: {exc}") from exc


def _profile(cfg: AppConfig, name: str | None) -> dict:
    selected = name or cfg.active_production_profile
    profiles = [
        p for p in cfg.production_profiles
        if isinstance(p, dict) and p.get("name") == selected
    ]
    if profiles:
        return profiles[0]
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


def _image_from_path(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot load image: {path}")
    return image


def _measurement_pixel_to_world(cfg: AppConfig, cad_points: list[dict], image_points: list[dict]):
    lc = cfg.lens_calibration
    if not lc.coordinate_correction or lc.correction_model_type not in {"homography", "affine"}:
        return None
    try:
        from cadviewer.calibration.coordinate_correction import CoordinateTransformer
        from cadviewer.registration import affine_solver
        transformer = CoordinateTransformer()
        if not transformer.load_model(lc.coordinate_correction, lc.correction_model_type):
            return None
        image_px = np.array([p["pixel"] for p in image_points], dtype=np.float64)
        from cadviewer.calibration.residual_map import residual_map_from_config
        residual_map = residual_map_from_config(cfg)
        if residual_map is not None:
            image_px = residual_map.correct(image_px)
        image_metric = transformer.transform(image_px)
        cad_metric = np.array([p["world"] for p in cad_points], dtype=np.float64)
        plane_to_world = affine_solver.solve_similarity(image_metric, cad_metric)
        if lc.correction_model_type == "homography":
            model_matrix = np.asarray(lc.coordinate_correction.get("homography"), dtype=np.float64)
        else:
            affine_model = np.asarray(lc.coordinate_correction.get("affine"), dtype=np.float64)
            model_matrix = np.vstack([affine_model, np.array([0.0, 0.0, 1.0])])
        if model_matrix.shape != (3, 3):
            return None
        measurement_transform = plane_to_world @ model_matrix
        metadata = (lc.coordinate_correction or {}).get("metadata", {})
        image_size = None
        if isinstance(metadata, dict):
            size = metadata.get("image_size")
            if isinstance(size, (list, tuple)) and len(size) == 2:
                image_size = (int(size[0]), int(size[1]))
        from cadviewer.calibration.transform_safety import validate_pixel_to_world_transform
        safety = validate_pixel_to_world_transform(
            measurement_transform, float(cfg.pixel_size_mm), image_size=image_size,
        )
        if not safety.safe:
            return None
        return measurement_transform
    except Exception:
        return None


def _auto_affine(image: np.ndarray, cfg: AppConfig, profile: dict, repo) -> tuple[np.ndarray, dict]:
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
    measurement_pixel_to_world = _measurement_pixel_to_world(cfg, cad_points, image_points)
    return affine, {
        "detections": [d.to_dict() for d in detections],
        "pixel_size_mm": pixel_size,
        "measurement_pixel_to_world": (
            measurement_pixel_to_world.tolist()
            if measurement_pixel_to_world is not None else None
        ),
        "measurement_transform_model": (
            cfg.lens_calibration.correction_model_type
            if measurement_pixel_to_world is not None else "affine"
        ),
    }


def _evaluate(
    repo, image: np.ndarray, affine: np.ndarray, pixel_size: float, query: str,
    pixel_to_world_transform=None, residual_map=None,
) -> list[dict]:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pipeline = MeasurementPipeline(
        repo, gray, affine, pixel_size_mm=pixel_size,
        residual_map=residual_map,
        pixel_to_world_transform=pixel_to_world_transform,
    )
    results = QueryEvaluator(repo, pipeline).evaluate(query)
    rows = []
    for result in results:
        inst = result.instruction
        rows.append({
            "query": inst.raw_text if inst else "",
            "status": result.status,
            "value": result.value,
            "nominal": result.nominal,
            "deviation": result.deviation,
        })
    return rows


def _print_stats(samples: list[list[dict]]) -> None:
    by_query: dict[str, list[float]] = {}
    for sample in samples:
        for row in sample:
            if row["value"] is not None:
                by_query.setdefault(row["query"], []).append(float(row["value"]))

    print("\nRepeatability summary (measured value, mm)")
    for query, values in by_query.items():
        if len(values) == 1:
            stdev = 0.0
        else:
            stdev = statistics.pstdev(values)
        print(
            f"{query:42s} n={len(values):2d} "
            f"mean={statistics.mean(values):.5f} "
            f"min={min(values):.5f} max={max(values):.5f} "
            f"range={max(values) - min(values):.5f} std={stdev:.5f}"
        )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dxf", default="xintai.dxf")
    parser.add_argument("--query", default="query2.txt")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--save-json", default="")
    args = parser.parse_args(argv)

    cfg = AppConfig.load()
    profile = _profile(cfg, args.profile)
    repo = DXFImporter().import_file(args.dxf)
    query = Path(args.query).read_text(encoding="utf-8")

    samples: list[list[dict]] = []
    details: list[dict] = []
    for idx in range(args.frames):
        if args.image:
            frame = _image_from_path(args.image[idx % len(args.image)])
        else:
            frame = _camera_frame(cfg, profile)
        image, undistorted = undistort_if_calibrated(frame, cfg)
        affine, reg = _auto_affine(image, cfg, profile, repo)
        measurement_transform = reg.get("measurement_pixel_to_world")
        if measurement_transform is not None:
            measurement_transform = np.array(measurement_transform, dtype=np.float64)
        from cadviewer.calibration.residual_map import residual_map_from_config
        rows = _evaluate(
            repo, image, affine, reg["pixel_size_mm"], query,
            pixel_to_world_transform=measurement_transform,
            residual_map=residual_map_from_config(cfg),
        )
        samples.append(rows)
        details.append({
            "frame": idx + 1,
            "undistorted": undistorted,
            "registration": reg,
            "results": rows,
        })
        compact = ", ".join(
            f"{r['value']:.4f}" if r["value"] is not None else r["status"]
            for r in rows
        )
        print(f"frame {idx + 1:02d}: {compact}")

    _print_stats(samples)
    if args.save_json:
        Path(args.save_json).write_text(
            json.dumps(details, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

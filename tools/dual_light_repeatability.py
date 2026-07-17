#!/usr/bin/env python3
"""Manual dual-light repeatability validation command.

The command opens the configured camera, prompts the operator to switch lights,
captures fresh software-triggered backlight/ring-light pairs, runs the shared
dual-light pipeline, and writes per-run plus summary CSV files.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QCoreApplication

from cadviewer.camera import HAS_CAMERA, CameraClass, CameraSettings
from cadviewer.core.config import AppConfig
from cadviewer.parsers.dxf_importer import DXFImporter
from cadviewer.registration.auto_correspondence import undistort_if_calibrated
from cadviewer.measurement.dual_light_pipeline import run_dual_light_measurement


def _active_profile(cfg: AppConfig) -> dict:
    name = str(getattr(cfg, "active_production_profile", "") or "")
    profiles = getattr(cfg, "production_profiles", []) or []
    for profile in profiles:
        if str(profile.get("name", "")) == name:
            return dict(profile)
    return dict(profiles[0]) if profiles else {}


def _camera_sections(profile: dict, cfg: AppConfig) -> dict:
    fallback = cfg.camera.__dict__.copy()
    camera = profile.get("camera", {}) if isinstance(profile, dict) else {}
    if isinstance(camera, dict):
        flat = {k: v for k, v in camera.items() if k in fallback}
        if flat:
            fallback.update(flat)
    sections = {
        "live_preview": dict(fallback),
        "backlight": dict(fallback),
        "ring_light": dict(fallback),
    }
    if isinstance(camera, dict):
        for key in sections:
            if isinstance(camera.get(key), dict):
                sections[key].update(camera[key])
    return sections


def _settings(data: dict) -> CameraSettings:
    allowed = CameraSettings.__dataclass_fields__.keys()
    return CameraSettings(**{k: data[k] for k in allowed if k in data})


def _trigger_capture(camera, app: QCoreApplication, timeout_ms: int = 3000):
    frame_box = {"frame": None}

    def _on_grab(frame):
        frame_box["frame"] = frame

    camera.signals.grab_done.connect(_on_grab)
    try:
        camera.software_trigger()
        deadline = time.monotonic() + timeout_ms / 1000.0
        while frame_box["frame"] is None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        if frame_box["frame"] is None:
            raise RuntimeError("Software trigger timed out")
        return frame_box["frame"]
    finally:
        try:
            camera.signals.grab_done.disconnect(_on_grab)
        except Exception:
            pass


def _capture_mode(camera, app, mode: str, settings: dict, settle_ms: int):
    camera.apply_settings(_settings(settings))
    app.processEvents()
    if mode == "backlight":
        print("Turn ON backlight and turn OFF ring light, then press Enter.")
    else:
        print("Turn OFF backlight and turn ON ring light, then press Enter.")
    input()
    if settle_ms > 0:
        time.sleep(settle_ms / 1000.0)
    return _trigger_capture(camera, app)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dxf", default="")
    parser.add_argument("--count", "-n", type=int, default=10)
    parser.add_argument("--output-dir", default="/tmp/cadviewer_dual_light_repeatability")
    args = parser.parse_args(argv)

    if not HAS_CAMERA:
        raise RuntimeError("Camera support is not available")
    cfg = AppConfig.load()
    profile = _active_profile(cfg)
    dxf = args.dxf or getattr(cfg, "last_dxf_path", "")
    if not dxf:
        raise RuntimeError("No DXF path supplied and AppConfig.last_dxf_path is empty")
    repo = DXFImporter().import_file(dxf)
    edge_ids = (profile.get("window_registration", {}) or {}).get("edge_ids", [])
    if len(edge_ids) != 4:
        raise RuntimeError("Active profile must contain exactly 4 window CAD edges")
    window_fit = profile.get("window_registration", {}) or {}
    fit_mode = "light-inner"
    try:
        light_fraction = float(window_fit.get("light_fraction", 0.95))
    except Exception:
        light_fraction = 0.95
    light_fraction = max(0.50, min(0.98, light_fraction))
    edge_bias = "inner"
    query_text = str(getattr(cfg, "measurement_queries", "") or "")
    if not query_text.strip():
        raise RuntimeError("No measurement queries configured")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sections = _camera_sections(profile, cfg)
    settle_ms = int((profile.get("capture", {}) or {}).get("settle_delay_ms", 200) or 0)

    app = QCoreApplication.instance() or QCoreApplication([])
    camera = CameraClass()
    devices = camera.enumerate_devices()
    if not devices:
        raise RuntimeError("No camera detected")
    camera.open(devices[0]["dev_info"])
    camera.set_trigger_mode()

    rows = []
    try:
        for index in range(args.count):
            print(f"Capture pair {index + 1}/{args.count}")
            backlight = _capture_mode(camera, app, "backlight", sections["backlight"], settle_ms)
            ring = _capture_mode(camera, app, "ring_light", sections["ring_light"], settle_ms)
            backlight, back_applied = undistort_if_calibrated(backlight, cfg)
            ring, ring_applied = undistort_if_calibrated(ring, cfg)
            result = run_dual_light_measurement(
                repo=repo,
                query_text=query_text,
                backlight_image=backlight,
                ring_light_image=ring,
                edge_tokens=edge_ids,
                pixel_size_mm=float(cfg.pixel_size_mm),
                line_fit_side_overrides=getattr(cfg, "line_fit_side_overrides", {}) or {},
                fit_mode=fit_mode,
                light_fraction=light_fraction,
                edge_bias=edge_bias,
                output_dir=output_dir / f"run_{index + 1:03d}",
                metadata={
                    "run_index": index + 1,
                    "backlight_undistorted": bool(back_applied),
                    "ring_light_undistorted": bool(ring_applied),
                    "settle_delay_ms": settle_ms,
                    "backlight_window_fit": {
                        "fit_mode": fit_mode,
                        "light_fraction": light_fraction,
                        "edge_bias": edge_bias,
                    },
                },
            )
            for result_row in result.results:
                inst = result_row.instruction
                rows.append({
                    "run": index + 1,
                    "query": inst.raw_text if inst else "",
                    "status": result_row.status,
                    "value": result_row.value,
                    "nominal": result_row.nominal,
                    "deviation": result_row.deviation,
                })
    finally:
        camera.close()

    csv_path = output_dir / "repeatability.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["run", "query", "status", "value", "nominal", "deviation"])
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for query in sorted({row["query"] for row in rows}):
        values = [float(row["value"]) for row in rows if row["query"] == query and row["value"] is not None]
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        summary.append({
            "query": query,
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "range": float(np.max(arr) - np.min(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        })
    summary_path = output_dir / "repeatability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {csv_path}")
    print(f"Saved {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

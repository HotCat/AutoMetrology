"""
Calibration Validation — measures known distances on the calibration grid
to determine whether TPS or measurement is responsible for scale errors.

Ignores CAD. Ignores registration. Ignores measurement.
Uses only calibration target data.

Known distances (10, 20, 50, 100 mm) are measured after:
  A. OpenCV undistortion only
  B. OpenCV undistortion + TPS correction
  C. Raw pixels (no correction)

If TPS introduces a systematic scale shift, distances B will differ from A.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

from ..calibration.residual_map import ResidualDistortionMap


@dataclass
class DistanceMeasurement:
    """Single distance measurement result."""

    known_distance_mm: float
    method: str  # "raw_pixels", "opencv_only", "opencv_tps"
    measured_px: float
    measured_mm: float
    absolute_error_mm: float
    relative_error_pct: float


@dataclass
class CalibrationValidationReport:
    """Report comparing known distances against measured distances."""

    measurements: List[DistanceMeasurement] = field(default_factory=list)
    pixel_size_mm: float = 0.0
    opencv_rms: float = 0.0
    tps_rms_before: float = 0.0
    tps_rms_after: float = 0.0

    def to_dict(self) -> dict:
        return {
            "pixel_size_mm": self.pixel_size_mm,
            "opencv_rms": self.opencv_rms,
            "tps_rms_before": self.tps_rms_before,
            "tps_rms_after": self.tps_rms_after,
            "measurements": [
                {
                    "known_mm": m.known_distance_mm,
                    "method": m.method,
                    "measured_px": m.measured_px,
                    "measured_mm": m.measured_mm,
                    "abs_error_mm": m.absolute_error_mm,
                    "rel_error_pct": m.relative_error_pct,
                }
                for m in self.measurements
            ],
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, default=_json_default))

    def summary(self) -> str:
        """Human-readable summary identifying the source of scale error."""
        lines = [
            "=" * 70,
            "CALIBRATION VALIDATION REPORT",
            "=" * 70,
            f"Pixel Size: {self.pixel_size_mm:.6f} mm/px",
            f"OpenCV RMS: {self.opencv_rms:.4f} px",
            f"TPS RMS Before: {self.tps_rms_before:.4f} px",
            f"TPS RMS After: {self.tps_rms_after:.4f} px",
            "",
            f"{'Known (mm)':>12} {'Method':<16} {'Meas (mm)':>10} "
            f"{'Error (mm)':>12} {'Error (%)':>10}",
            "-" * 62,
        ]

        for m in self.measurements:
            lines.append(
                f"{m.known_distance_mm:>12.3f} {m.method:<16} {m.measured_mm:>10.4f} "
                f"{m.absolute_error_mm:>+12.4f} {m.relative_error_pct:>+10.2f}%"
            )

        # Analysis: group by method
        lines.extend(["", "-" * 62, "ANALYSIS"])

        methods = {}
        for m in self.measurements:
            if m.method not in methods:
                methods[m.method] = []
            methods[m.method].append(m.relative_error_pct)

        for method, errors in methods.items():
            mean_err = sum(errors) / len(errors)
            max_err = max(abs(e) for e in errors)
            lines.append(f"  {method}: mean error = {mean_err:+.3f}%, max = {max_err:.3f}%")

        # Root cause analysis
        lines.extend(["", "ROOT CAUSE ANALYSIS"])

        raw_errs = methods.get("raw_pixels", [])
        cv_errs = methods.get("opencv_only", [])
        tps_errs = methods.get("opencv_tps", [])

        if raw_errs and cv_errs:
            raw_mean = abs(sum(raw_errs) / len(raw_errs))
            cv_mean = abs(sum(cv_errs) / len(cv_errs))
            if cv_mean > raw_mean * 1.5:
                lines.append("  WARNING: OpenCV undistortion INCREASES distance error.")
                lines.append("  The distortion model may be overfitting or incorrect.")
            else:
                lines.append("  OpenCV undistortion does not significantly affect distances.")

        if cv_errs and tps_errs:
            cv_mean = abs(sum(cv_errs) / len(cv_errs))
            tps_mean = abs(sum(tps_errs) / len(tps_errs))
            if tps_mean > cv_mean * 1.5:
                lines.append("  WARNING: TPS correction INTRODUCES a scale shift.")
                lines.append("  TPS residuals likely contain a scale component.")
                diff = tps_mean - cv_mean
                lines.append(f"  Estimated TPS-induced scale error: ~{diff:.2f}%")
            else:
                lines.append("  TPS correction does not significantly affect distances.")

        lines.append("=" * 70)
        return "\n".join(lines)


class CalibrationValidator:
    """Validates calibration quality by measuring known grid distances."""

    def __init__(
        self,
        corners_raw: np.ndarray,
        cols: int,
        rows: int,
        cell_mm: float,
        pixel_size_mm: float,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        residual_map: Optional[ResidualDistortionMap] = None,
        opencv_rms: float = 0.0,
    ) -> None:
        self._corners_raw = corners_raw  # Nx2 detected corners in raw image
        self._cols = cols
        self._rows = rows
        self._cell_mm = cell_mm
        self._pixel_size_mm = pixel_size_mm
        self._camera_matrix = camera_matrix
        self._dist_coeffs = dist_coeffs
        self._residual_map = residual_map
        self._opencv_rms = opencv_rms

    def validate(self) -> CalibrationValidationReport:
        """Run validation and return report."""

        report = CalibrationValidationReport(
            pixel_size_mm=self._pixel_size_mm,
            opencv_rms=self._opencv_rms,
        )

        # Prepare corner sets: raw, undistorted, undistorted+TPS
        corners_opencv = self._corners_raw.copy()
        if self._camera_matrix is not None and self._dist_coeffs is not None:
            try:
                import cv2
                corners_opencv = cv2.undistortPoints(
                    self._corners_raw.reshape(-1, 1, 2).astype(np.float32),
                    self._camera_matrix,
                    self._dist_coeffs,
                    P=self._camera_matrix,
                ).reshape(-1, 2)
            except Exception:
                corners_opencv = self._corners_raw.copy()

        corners_tps = corners_opencv.copy()
        if self._residual_map is not None and self._residual_map.is_built:
            corners_tps = self._residual_map.correct(corners_opencv)

        # Compute TPS RMS
        if self._residual_map is not None and self._residual_map.is_built:
            # The ideal grid positions
            ideal = self._ideal_grid(corners_opencv)
            report.tps_rms_before = float(np.sqrt(
                np.mean(np.sum((corners_opencv - ideal) ** 2, axis=1))
            ))
            report.tps_rms_after = float(np.sqrt(
                np.mean(np.sum((corners_tps - ideal) ** 2, axis=1))
            ))

        # Measure known distances at multiple scales
        for known_dist in [10.0, 20.0, 50.0, 100.0]:
            n_cells = known_dist / self._cell_mm

            if abs(n_cells - round(n_cells)) > 0.01:
                continue  # skip distances not aligned with grid

            n_cells = int(round(n_cells))
            if n_cells < 1 or n_cells >= self._cols or n_cells >= self._rows:
                continue

            # Horizontal distances: measure between columns separated by n_cells
            for method_name, corners in [
                ("raw_pixels", self._corners_raw),
                ("opencv_only", corners_opencv),
                ("opencv_tps", corners_tps),
            ]:
                dists = []
                for row in range(self._rows):
                    for col_start in range(self._cols - n_cells):
                        col_end = col_start + n_cells
                        idx_start = row * self._cols + col_start
                        idx_end = row * self._cols + col_end
                        if idx_end < len(corners):
                            dx = corners[idx_end, 0] - corners[idx_start, 0]
                            dy = corners[idx_end, 1] - corners[idx_start, 1]
                            dist_px = math.sqrt(dx * dx + dy * dy)
                            dists.append(dist_px)

                if dists:
                    mean_px = sum(dists) / len(dists)
                    mean_mm = mean_px * self._pixel_size_mm
                    abs_err = mean_mm - known_dist
                    rel_err = abs_err / known_dist * 100 if known_dist > 0 else 0

                    report.measurements.append(DistanceMeasurement(
                        known_distance_mm=known_dist,
                        method=method_name,
                        measured_px=mean_px,
                        measured_mm=mean_mm,
                        absolute_error_mm=abs_err,
                        relative_error_pct=rel_err,
                    ))

        return report

    def _ideal_grid(self, detected: np.ndarray) -> np.ndarray:
        """Compute ideal grid positions via affine fit to detected corners."""
        n = self._cols * self._rows
        if len(detected) < n:
            return detected.copy()

        grid = np.zeros((n, 2), dtype=np.float64)
        for r in range(self._rows):
            for c in range(self._cols):
                grid[r * self._cols + c] = [c, r]

        A = np.column_stack([grid, np.ones(n)])
        coeff_x, _, _, _ = np.linalg.lstsq(A, detected[:n, 0], rcond=None)
        coeff_y, _, _, _ = np.linalg.lstsq(A, detected[:n, 1], rcond=None)

        ideal_x = A @ coeff_x
        ideal_y = A @ coeff_y
        return np.column_stack([ideal_x, ideal_y])


def run_calibration_validation(
    corners_raw: np.ndarray,
    cols: int,
    rows: int,
    cell_mm: float,
    pixel_size_mm: float,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
    residual_map: Optional[ResidualDistortionMap] = None,
    opencv_rms: float = 0.0,
    output_path: Optional[Path] = None,
) -> CalibrationValidationReport:
    """Run calibration validation and optionally save report."""

    validator = CalibrationValidator(
        corners_raw, cols, rows, cell_mm, pixel_size_mm,
        camera_matrix, dist_coeffs, residual_map, opencv_rms,
    )
    report = validator.validate()

    if output_path is not None:
        report.save(output_path)
        print(f"[CALIB] Report saved to {output_path}")

    return report


def _json_default(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
"""
Reference Test — verification measurement using calibration plate data.

Selects two known corners from the calibration grid and measures the
distance between them under three configurations:

  1. Original: raw pixels + pixel_size_mm only (no correction)
  2. OpenCV Only: undistorted pixels + pixel_size_mm
  3. OpenCV + TPS: undistorted pixels + TPS correction + pixel_size_mm

This isolates whether TPS introduces scale distortion.

If TPS changes the distance, it directly proves TPS is the source of
the measurement error.
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
class ReferenceTestResult:
    """Result of a single reference distance test."""

    test_name: str
    known_distance_mm: float
    method: str  # "raw", "opencv_only", "opencv_tps"
    pixel_distance: float
    mm_distance: float
    absolute_error_mm: float
    relative_error_pct: float


@dataclass
class ReferenceTestReport:
    """Complete report comparing three correction methods on known distances."""

    results: List[ReferenceTestResult] = field(default_factory=list)
    pixel_size_mm: float = 0.0
    chessboard_cols: int = 0
    chessboard_rows: int = 0
    cell_mm: float = 0.0

    # Diagnosis
    raw_mean_error_pct: float = 0.0
    opencv_mean_error_pct: float = 0.0
    tps_mean_error_pct: float = 0.0

    diagnosis: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pixel_size_mm": self.pixel_size_mm,
            "chessboard": {"cols": self.chessboard_cols, "rows": self.chessboard_rows, "cell_mm": self.cell_mm},
            "results": [
                {
                    "test": r.test_name, "method": r.method,
                    "known_mm": r.known_distance_mm,
                    "px_dist": r.pixel_distance,
                    "mm_dist": r.mm_distance,
                    "abs_err_mm": r.absolute_error_mm,
                    "rel_err_pct": r.relative_error_pct,
                }
                for r in self.results
            ],
            "summary": {
                "raw_mean_error_pct": self.raw_mean_error_pct,
                "opencv_mean_error_pct": self.opencv_mean_error_pct,
                "tps_mean_error_pct": self.tps_mean_error_pct,
            },
            "diagnosis": self.diagnosis,
            "warnings": self.warnings,
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, default=_json_default))

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "REFERENCE TEST REPORT",
            "=" * 70,
            f"Pixel Size: {self.pixel_size_mm:.6f} mm/px",
            f"Chessboard: {self.chessboard_cols}x{self.chessboard_rows}, cell={self.cell_mm} mm",
            "",
            f"{'Test':<25} {'Method':<16} {'Known':>8} {'Measured':>10} "
            f"{'Error':>10} {'Err%':>8}",
            "-" * 80,
        ]

        for r in self.results:
            lines.append(
                f"{r.test_name:<25} {r.method:<16} {r.known_distance_mm:>8.3f} "
                f"{r.mm_distance:>10.4f} {r.absolute_error_mm:>+10.4f} "
                f"{r.relative_error_pct:>+8.3f}%"
            )

        lines.extend([
            "",
            "-" * 80,
            "SUMMARY:",
            f"  Raw pixels mean error:     {self.raw_mean_error_pct:+.4f}%",
            f"  OpenCV only mean error:    {self.opencv_mean_error_pct:+.4f}%",
            f"  OpenCV + TPS mean error:   {self.tps_mean_error_pct:+.4f}%",
            "",
        ])

        if self.warnings:
            lines.append("WARNINGS:")
            for w in self.warnings:
                lines.append(f"  * {w}")
            lines.append("")

        if self.diagnosis:
            lines.extend(["DIAGNOSIS:", f"  {self.diagnosis}", ""])

        lines.append("=" * 70)
        return "\n".join(lines)


class ReferenceTester:
    """Runs reference distance tests using calibration grid data."""

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
    ) -> None:
        self._corners_raw = corners_raw
        self._cols = cols
        self._rows = rows
        self._cell_mm = cell_mm
        self._pixel_size_mm = pixel_size_mm
        self._camera_matrix = camera_matrix
        self._dist_coeffs = dist_coeffs
        self._residual_map = residual_map

    def run(self) -> ReferenceTestReport:
        """Run all reference tests."""
        report = ReferenceTestReport(
            pixel_size_mm=self._pixel_size_mm,
            chessboard_cols=self._cols,
            chessboard_rows=self._rows,
            cell_mm=self._cell_mm,
        )

        # Prepare corner sets for each method
        corners_raw = self._corners_raw.copy()

        # OpenCV undistorted
        corners_cv = self._corners_raw.copy()
        if self._camera_matrix is not None and self._dist_coeffs is not None:
            try:
                import cv2
                corners_cv = cv2.undistortPoints(
                    self._corners_raw.reshape(-1, 1, 2).astype(np.float32),
                    self._camera_matrix,
                    self._dist_coeffs,
                    P=self._camera_matrix,
                ).reshape(-1, 2)
            except Exception:
                pass

        # OpenCV + TPS
        corners_tps = corners_cv.copy()
        if self._residual_map is not None and self._residual_map.is_built:
            corners_tps = self._residual_map.correct(corners_cv)

        # Test configurations: (test_name, corners, method)
        test_configs = [
            ("raw", corners_raw, "raw"),
            ("opencv", corners_cv, "opencv_only"),
            ("opencv+tps", corners_tps, "opencv_tps"),
        ]

        # Pick a known-distance test: corners separated by N cells
        # Test at multiple scales
        test_distances = []
        for n_cells in [1, 2, 3, 5]:
            known = n_cells * self._cell_mm
            if n_cells < self._cols:
                test_distances.append((n_cells, known))

        for n_cells, known_mm in test_distances:
            for test_name, corners, method in test_configs:
                # Measure: horizontal distance between (col=0, row=mid) and (col=n_cells, row=mid)
                mid_row = self._rows // 2

                idx_start = mid_row * self._cols + 0
                idx_end = mid_row * self._cols + n_cells

                if idx_end < len(corners):
                    dx = corners[idx_end, 0] - corners[idx_start, 0]
                    dy = corners[idx_end, 1] - corners[idx_start, 1]
                    dist_px = math.sqrt(dx * dx + dy * dy)
                    dist_mm = dist_px * self._pixel_size_mm
                    abs_err = dist_mm - known_mm
                    rel_err = abs_err / known_mm * 100 if known_mm > 0 else 0

                    report.results.append(ReferenceTestResult(
                        test_name=f"{n_cells}_cells_horizontal",
                        known_distance_mm=known_mm,
                        method=method,
                        pixel_distance=dist_px,
                        mm_distance=dist_mm,
                        absolute_error_mm=abs_err,
                        relative_error_pct=rel_err,
                    ))

        # Compute mean errors per method
        method_errors = {"raw": [], "opencv_only": [], "opencv_tps": []}
        for r in report.results:
            if r.method in method_errors:
                method_errors[r.method].append(r.relative_error_pct)

        report.raw_mean_error_pct = (
            sum(method_errors["raw"]) / len(method_errors["raw"])
            if method_errors["raw"] else 0
        )
        report.opencv_mean_error_pct = (
            sum(method_errors["opencv_only"]) / len(method_errors["opencv_only"])
            if method_errors["opencv_only"] else 0
        )
        report.tps_mean_error_pct = (
            sum(method_errors["opencv_tps"]) / len(method_errors["opencv_tps"])
            if method_errors["opencv_tps"] else 0
        )

        # Diagnosis
        self._generate_diagnosis(report)

        return report

    def _generate_diagnosis(self, report: ReferenceTestReport) -> None:
        """Generate root-cause diagnosis."""

        raw = report.raw_mean_error_pct
        cv = report.opencv_mean_error_pct
        tps = report.tps_mean_error_pct

        # If raw and cv are similar, but tps differs significantly → TPS is the issue
        if abs(raw - cv) < 0.3 and abs(tps - cv) > 0.5:
            direction = "increases" if abs(tps) > abs(cv) else "decreases"
            report.diagnosis = (
                f"TPS {direction} measurement error. "
                f"Raw: {raw:+.3f}%, OpenCV: {cv:+.3f}%, OpenCV+TPS: {tps:+.3f}%. "
                "Since raw and OpenCV-only are consistent but TPS differs, "
                "TPS is introducing a systematic error. The TPS residual map "
                "was likely built from a subset of corners that biased the "
                "interpolation toward a different scale."
            )
        elif abs(cv - raw) > 0.5:
            report.diagnosis = (
                f"OpenCV undistortion itself changes scale. "
                f"Raw: {raw:+.3f}%, OpenCV: {cv:+.3f}%. "
                "The distortion model may be over-fitted or the calibration "
                "images may have insufficient coverage."
            )
        elif abs(tps) < 0.5 and abs(raw) < 0.5 and abs(cv) < 0.5:
            report.diagnosis = (
                "All three methods produce accurate results on the calibration grid. "
                "The measurement error is NOT in the calibration/TPS — it must be "
                "elsewhere (registration, feature fitting, or coordinate system mixing)."
            )
        else:
            report.diagnosis = (
                "Mixed results — see individual test rows for details."
            )

        # Warnings
        if abs(tps - cv) > 1.0:
            report.warnings.append(
                f"TPS introduces {tps - cv:+.3f}% error compared to OpenCV-only. "
                "This is significant and will affect all measurements."
            )

        if abs(raw) > 1.0:
            report.warnings.append(
                f"Raw pixel measurement has {raw:+.3f}% error. "
                "pixel_size_mm calibration itself may be inaccurate."
            )


def run_reference_test(
    corners_raw: np.ndarray,
    cols: int,
    rows: int,
    cell_mm: float,
    pixel_size_mm: float,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
    residual_map: Optional[ResidualDistortionMap] = None,
    output_path: Optional[Path] = None,
) -> ReferenceTestReport:
    """Run reference test and optionally save report."""

    tester = ReferenceTester(
        corners_raw, cols, rows, cell_mm, pixel_size_mm,
        camera_matrix, dist_coeffs, residual_map,
    )
    report = tester.run()

    if output_path is not None:
        report.save(output_path)
        print(f"[REFTEST] Report saved to {output_path}")

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

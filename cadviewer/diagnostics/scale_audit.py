"""
Scale Calibration Audit — determines where pixel/mm conversion is calculated
and verifies whether it was computed before or after TPS.

CRITICAL BUG PATTERN:
  pixel_size_mm calibrated BEFORE TPS (on raw/undistorted pixels)
  measurement occurs AFTER TPS (on corrected pixels)
  → TPS changes effective pixel spacing → scale error in mm

This audit traces the scale calibration path and the measurement path,
then compares them to detect this mismatch.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

from ..calibration.residual_map import ResidualDistortionMap
from ..registration import affine_solver


@dataclass
class ScaleSource:
    """Describes where a scale value comes from."""

    name: str
    value_mm_per_px: float
    source: str  # e.g. "config.pixel_size_mm", "affine[0,0]", "grid_cell/pixel_count"
    computed_in: str  # "raw_pixel_space", "undistorted_pixel_space", "tps_corrected_space"


@dataclass
class ScaleAuditReport:
    """Complete scale calibration audit report."""

    # All sources of scale in the system
    scale_sources: List[ScaleSource] = field(default_factory=list)

    # The actual scale used for measurement
    measurement_scale_mm_per_px: float = 0.0
    measurement_scale_source: str = ""

    # Scale at which calibration was performed
    calibration_scale_mm_per_px: float = 0.0
    calibration_space: str = ""  # "raw", "undistorted", "tps_corrected"

    # Scale used by the affine transform
    affine_scale_mm_per_px: float = 0.0

    # Consistency checks
    scale_consistent: bool = False
    scale_error_pct: float = 0.0

    # Detailed analysis of the measurement pipeline
    pipeline_analysis: List[str] = field(default_factory=list)

    # Warnings
    warnings: List[str] = field(default_factory=list)

    # Diagnosis
    diagnosis: str = ""

    def to_dict(self) -> dict:
        return {
            "scale_sources": [
                {"name": s.name, "value": s.value_mm_per_px,
                 "source": s.source, "computed_in": s.computed_in}
                for s in self.scale_sources
            ],
            "measurement_scale": self.measurement_scale_mm_per_px,
            "measurement_scale_source": self.measurement_scale_source,
            "calibration_scale": self.calibration_scale_mm_per_px,
            "calibration_space": self.calibration_space,
            "affine_scale": self.affine_scale_mm_per_px,
            "scale_consistent": self.scale_consistent,
            "scale_error_pct": self.scale_error_pct,
            "pipeline_analysis": self.pipeline_analysis,
            "warnings": self.warnings,
            "diagnosis": self.diagnosis,
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, default=_json_default))

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "SCALE CALIBRATION AUDIT REPORT",
            "=" * 70,
            "",
            "SCALE SOURCES IN SYSTEM:",
        ]

        for s in self.scale_sources:
            lines.append(
                f"  {s.name:<30} = {s.value_mm_per_px:.6f} mm/px  "
                f"(from {s.source}, space: {s.computed_in})"
            )

        lines.extend([
            "",
            "MEASUREMENT SCALE:",
            f"  Value: {self.measurement_scale_mm_per_px:.6f} mm/px",
            f"  Source: {self.measurement_scale_source}",
            "",
            "CALIBRATION SCALE:",
            f"  Value: {self.calibration_scale_mm_per_px:.6f} mm/px",
            f"  Space: {self.calibration_space}",
            "",
            "AFFINE SCALE:",
            f"  Value: {self.affine_scale_mm_per_px:.6f} mm/px",
            "",
            f"CONSISTENT: {self.scale_consistent}",
            f"SCALE ERROR: {self.scale_error_pct:+.4f}%",
            "",
        ])

        if self.pipeline_analysis:
            lines.append("PIPELINE ANALYSIS:")
            for step in self.pipeline_analysis:
                lines.append(f"  {step}")
            lines.append("")

        if self.warnings:
            lines.append("WARNINGS:")
            for w in self.warnings:
                lines.append(f"  * {w}")
            lines.append("")

        if self.diagnosis:
            lines.extend([
                "DIAGNOSIS:",
                f"  {self.diagnosis}",
            ])

        lines.append("=" * 70)
        return "\n".join(lines)


class ScaleAuditor:
    """Audits scale calibration consistency across the measurement pipeline."""

    def __init__(
        self,
        pixel_size_mm: float,
        affine: np.ndarray,
        residual_map: Optional[ResidualDistortionMap] = None,
        image_undistorted: bool = False,
        chessboard_cell_mm: float = 0.0,
        corners_raw: Optional[np.ndarray] = None,
        cols: int = 0,
        rows: int = 0,
    ) -> None:
        self._pixel_size_mm = pixel_size_mm
        self._affine = affine
        self._residual_map = residual_map
        self._image_undistorted = image_undistorted
        self._chessboard_cell_mm = chessboard_cell_mm
        self._corners_raw = corners_raw
        self._cols = cols
        self._rows = rows

    def audit(self) -> ScaleAuditReport:
        report = ScaleAuditReport()

        # Source 1: config.pixel_size_mm
        report.scale_sources.append(ScaleSource(
            name="config.pixel_size_mm",
            value_mm_per_px=self._pixel_size_mm,
            source="AppConfig.pixel_size_mm",
            computed_in="calibration_space",  # unknown until we trace it
        ))
        report.measurement_scale_mm_per_px = self._pixel_size_mm
        report.measurement_scale_source = "config.pixel_size_mm"

        # Source 2: Affine scale (from registration)
        params = affine_solver.extract_params(self._affine)
        affine_scale = params['scale_x']
        report.scale_sources.append(ScaleSource(
            name="affine_scale",
            value_mm_per_px=affine_scale,
            source="affine_solver.extract_params(T)['scale_x']",
            computed_in="registration_space",
        ))
        report.affine_scale_mm_per_px = affine_scale

        # Source 3: Empirical scale from chessboard (if data available)
        if (self._corners_raw is not None and self._cols > 0 and self._rows > 0
                and self._chessboard_cell_mm > 0):
            empirical_scale = self._compute_empirical_scale()
            if empirical_scale > 0:
                space = "raw_pixel_space"
                report.scale_sources.append(ScaleSource(
                    name="empirical_chessboard_raw",
                    value_mm_per_px=empirical_scale,
                    source=f"chessboard_cell_mm / mean_inter_corner_px ({self._chessboard_cell_mm:.1f}mm grid)",
                    computed_in=space,
                ))
                report.calibration_scale_mm_per_px = empirical_scale
                report.calibration_space = space

            # Also compute scale after TPS
            if self._residual_map is not None and self._residual_map.is_built:
                tps_scale = self._compute_empirical_scale_after_tps()
                if tps_scale > 0:
                    report.scale_sources.append(ScaleSource(
                        name="empirical_chessboard_tps",
                        value_mm_per_px=tps_scale,
                        source=f"chessboard_cell_mm / mean_inter_corner_px_after_tps",
                        computed_in="tps_corrected_space",
                    ))

        # Pipeline analysis
        report.pipeline_analysis = self._analyze_pipeline()

        # Consistency check
        scales = [s.value_mm_per_px for s in report.scale_sources]
        if len(scales) >= 2:
            max_scale = max(scales)
            min_scale = min(scales)
            if max_scale > 0:
                report.scale_error_pct = (max_scale - min_scale) / max_scale * 100
                report.scale_consistent = report.scale_error_pct < 0.5  # < 0.5% tolerance

        # Warnings and diagnosis
        self._generate_warnings(report)
        self._generate_diagnosis(report)

        return report

    def _compute_empirical_scale(self) -> float:
        """Compute mm/px from raw chessboard corner spacing."""
        corners = self._corners_raw
        if corners is None or len(corners) < self._cols:
            return 0.0

        # Horizontal inter-corner distances
        h_dists = []
        for row in range(self._rows):
            for col in range(self._cols - 1):
                idx1 = row * self._cols + col
                idx2 = row * self._cols + col + 1
                if idx2 < len(corners):
                    dx = corners[idx2, 0] - corners[idx1, 0]
                    dy = corners[idx2, 1] - corners[idx1, 1]
                    h_dists.append(math.sqrt(dx * dx + dy * dy))

        if not h_dists:
            return 0.0

        mean_px = sum(h_dists) / len(h_dists)
        return self._chessboard_cell_mm / mean_px if mean_px > 0 else 0.0

    def _compute_empirical_scale_after_tps(self) -> float:
        """Compute mm/px from chessboard corner spacing AFTER TPS correction."""
        if self._residual_map is None or not self._residual_map.is_built:
            return 0.0

        corners = self._residual_map.correct(self._corners_raw)

        h_dists = []
        for row in range(self._rows):
            for col in range(self._cols - 1):
                idx1 = row * self._cols + col
                idx2 = row * self._cols + col + 1
                if idx2 < len(corners):
                    dx = corners[idx2, 0] - corners[idx1, 0]
                    dy = corners[idx2, 1] - corners[idx1, 1]
                    h_dists.append(math.sqrt(dx * dx + dy * dy))

        if not h_dists:
            return 0.0

        mean_px = sum(h_dists) / len(h_dists)
        return self._chessboard_cell_mm / mean_px if mean_px > 0 else 0.0

    def _analyze_pipeline(self) -> List[str]:
        """Trace the pixel→mm conversion path through the measurement code."""
        steps = [
            "1. Pixel size calibration: chessboard corners → cell_mm / inter_corner_px",
            "   → stored as config.pixel_size_mm",
            "",
            "2. Image loading: ImageLoadDialog loads image",
            "   → if lens calibrated: cv2.undistort() applied to image",
            "   → pixel_size_mm set from config",
            "",
            "3. Registration: strategy.py converts image pixels to world mm",
            "   → img_world[:,0] *= pixel_size_mm  (img_contour_world)",
            "   → img_world[:,1] *= -pixel_size_mm  (Y-flip)",
            "   → Registration affine T maps CAD→image_world",
            "   → Image layer affine = inv(T) @ diag(pixel_size, -pixel_size, 1)",
            "   → This affine maps pixel→CAD_world",
            "",
            "4. Measurement Pipeline: MeasurementPipeline.__init__(affine=...)",
            "   → affine is the image_layer.affine (pixel→CAD_world)",
            "   → _correct_points() applies TPS correction to fitted pixel coords",
            "   → affine_solver.apply(affine, corrected_points) → world mm",
            "",
            "5. Query Evaluator: computes distances in world mm",
            "   → uses fitted_geometry_world directly",
        ]
        return steps

    def _generate_warnings(self, report: ScaleAuditReport) -> None:
        """Generate warnings about scale inconsistencies."""

        # Check: affine scale vs pixel_size_mm
        scale_diff = abs(report.affine_scale_mm_per_px - self._pixel_size_mm)
        if scale_diff > 0.0005:
            pct = scale_diff / self._pixel_size_mm * 100
            report.warnings.append(
                f"Affine scale ({report.affine_scale_mm_per_px:.6f}) differs from "
                f"pixel_size_mm ({self._pixel_size_mm:.6f}) by {pct:.3f}%. "
                "Registration and calibration disagree on the scale."
            )

        # Check: empirical scale vs config scale
        empirical_sources = [
            s for s in report.scale_sources if s.name.startswith("empirical_")
        ]
        for es in empirical_sources:
            diff_pct = (es.value_mm_per_px - self._pixel_size_mm) / self._pixel_size_mm * 100
            if abs(diff_pct) > 0.5:
                report.warnings.append(
                    f"Empirical scale from {es.name} ({es.value_mm_per_px:.6f}) "
                    f"differs from config ({self._pixel_size_mm:.6f}) by {diff_pct:+.3f}%. "
                    f"Computed in {es.computed_in}."
                )

        # Check: scale before TPS vs after TPS
        raw_scales = [s for s in report.scale_sources if "raw" in s.computed_in]
        tps_scales = [s for s in report.scale_sources if "tps" in s.computed_in]

        if raw_scales and tps_scales:
            for rs in raw_scales:
                for ts in tps_scales:
                    diff_pct = (ts.value_mm_per_px - rs.value_mm_per_px) / rs.value_mm_per_px * 100
                    if abs(diff_pct) > 0.5:
                        report.warnings.append(
                            f"TPS changes effective scale from {rs.value_mm_per_px:.6f} "
                            f"to {ts.value_mm_per_px:.6f} ({diff_pct:+.3f}%). "
                            "If pixel_size_mm was calibrated before TPS but measurement "
                            "uses TPS-corrected coordinates, this produces a systematic error."
                        )

    def _generate_diagnosis(self, report: ScaleAuditReport) -> None:
        """Generate root-cause diagnosis."""

        # Check the critical bug pattern
        raw_scales = [s for s in report.scale_sources if "raw" in s.computed_in]
        tps_scales = [s for s in report.scale_sources if "tps" in s.computed_in]

        if raw_scales and tps_scales:
            raw_mean = np.mean([s.value_mm_per_px for s in raw_scales])
            tps_mean = np.mean([s.value_mm_per_px for s in tps_scales])
            diff_pct = (tps_mean - raw_mean) / raw_mean * 100

            if abs(diff_pct) > 1.0:
                report.diagnosis = (
                    f"TPS introduces a {diff_pct:+.2f}% scale shift. "
                    f"Raw pixel scale: {raw_mean:.6f} mm/px, "
                    f"TPS-corrected scale: {tps_mean:.6f} mm/px. "
                    "The pixel_size_mm was calibrated using inter-corner spacing in "
                    "raw (or undistorted) pixel space. After TPS correction, the "
                    "effective pixel spacing changes. Measurement uses TPS-corrected "
                    "coordinates but the pixel_size_mm (or affine) was not recalibrated "
                    "in TPS-corrected space. FIX: either (A) recompute pixel_size_mm "
                    "after TPS correction, or (B) apply TPS correction to world "
                    "coordinates instead of pixel coordinates."
                )
                return

        # Check affine vs pixel_size_mm
        scale_diff = abs(report.affine_scale_mm_per_px - self._pixel_size_mm)
        if scale_diff > 0.001:
            pct = scale_diff / self._pixel_size_mm * 100
            report.diagnosis = (
                f"Registration affine scale ({report.affine_scale_mm_per_px:.6f}) "
                f"disagrees with pixel_size_mm ({self._pixel_size_mm:.6f}) by {pct:.3f}%. "
                "The affine transform maps pixels to CAD world coordinates. If the "
                "registration found a slightly different scale than the calibrated "
                "pixel_size_mm, measurements will have a systematic scale error. "
                "This is expected if registration is imperfect, but a >0.5% error "
                "suggests a problem."
            )
            return

        if report.scale_consistent:
            report.diagnosis = (
                "All scale sources are consistent within 0.5%. "
                "The scale calibration does not appear to be the source of the error."
            )
        else:
            report.diagnosis = (
                "Scale sources are inconsistent, but the specific cause "
                "needs manual investigation. See warnings above."
            )


def run_scale_audit(
    pixel_size_mm: float,
    affine: np.ndarray,
    residual_map: Optional[ResidualDistortionMap] = None,
    image_undistorted: bool = False,
    chessboard_cell_mm: float = 0.0,
    corners_raw: Optional[np.ndarray] = None,
    cols: int = 0,
    rows: int = 0,
    output_path: Optional[Path] = None,
) -> ScaleAuditReport:
    """Run scale calibration audit and optionally save report."""

    auditor = ScaleAuditor(
        pixel_size_mm, affine, residual_map, image_undistorted,
        chessboard_cell_mm, corners_raw, cols, rows,
    )
    report = auditor.audit()

    if output_path is not None:
        report.save(output_path)
        print(f"[SCALE] Report saved to {output_path}")

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

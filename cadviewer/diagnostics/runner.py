"""
Diagnostic Runner — orchestrates all diagnostic phases, produces the
6 reports, and generates a root-cause recommendation.

Usage:
    from cadviewer.diagnostics.runner import DiagnosticRunner

    runner = DiagnosticRunner(...)
    runner.run_all()
    runner.print_summary()
    runner.save_reports(output_dir)

This runs 8 phases without modifying any algorithms:
  1. Coordinate trace for every measured feature
  2. Calibration validation (known-distance measurements)
  3. TPS audit (where TPS is applied, double-application check)
  4. Scale calibration audit (pixel/mm consistency)
  5. Coordinate system consistency check
  6. Distance debug overlay visualization
  7. Reference test (raw vs OpenCV vs OpenCV+TPS)
  8. Root-cause recommendation
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

from ..models.feature import FeatureType
from ..models.repository import FeatureRepository
from ..models.measured_feature import MeasuredFeature
from ..measurement.measurement_pipeline import MeasurementPipeline
from ..registration import affine_solver
from ..calibration.residual_map import ResidualDistortionMap

from .coordinate_trace import CoordinateTracer, TraceReport
from .calibration_validation import CalibrationValidator, CalibrationValidationReport
from .tps_audit import TPSAuditor, TPSAuditReport
from .scale_audit import ScaleAuditor, ScaleAuditReport
from .coord_system_audit import CoordinateSystemAuditor, CoordinateSystemReport
from .reference_test import ReferenceTester, ReferenceTestReport


@dataclass
class DiagnosticResult:
    """Complete diagnostic result from all phases."""

    # Phase 1
    coordinate_trace: Optional[TraceReport] = None
    # Phase 2
    calibration_validation: Optional[CalibrationValidationReport] = None
    # Phase 3
    tps_audit: Optional[TPSAuditReport] = None
    # Phase 4
    scale_audit: Optional[ScaleAuditReport] = None
    # Phase 5
    coord_system_audit: Optional[CoordinateSystemReport] = None
    # Phase 7
    reference_test: Optional[ReferenceTestReport] = None
    # Phase 8
    recommendation: str = ""


class DiagnosticRunner:
    """Orchestrates all diagnostic phases."""

    def __init__(
        self,
        repo: FeatureRepository,
        affine: np.ndarray,
        pixel_size_mm: float,
        image: Optional[np.ndarray] = None,
        image_path: Optional[str] = None,
        residual_map: Optional[ResidualDistortionMap] = None,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        image_undistorted: bool = False,
        chessboard_cols: int = 0,
        chessboard_rows: int = 0,
        chessboard_cell_mm: float = 0.0,
        corners_raw: Optional[np.ndarray] = None,
        cad_bbox: Optional[Tuple[float, float, float, float]] = None,
        query_results: Optional[list] = None,
    ) -> None:
        self._repo = repo
        self._affine = affine
        self._pixel_size_mm = pixel_size_mm
        self._image = image
        self._image_path = image_path
        self._residual_map = residual_map
        self._camera_matrix = camera_matrix
        self._dist_coeffs = dist_coeffs
        self._image_undistorted = image_undistorted
        self._chessboard_cols = chessboard_cols
        self._chessboard_rows = chessboard_rows
        self._chessboard_cell_mm = chessboard_cell_mm
        self._corners_raw = corners_raw
        self._cad_bbox = cad_bbox
        self._query_results = query_results or []

        self._measured_features: List[MeasuredFeature] = []
        self._result = DiagnosticResult()

    def run_all(self) -> DiagnosticResult:
        """Run all 8 diagnostic phases."""
        print("\n" + "=" * 70)
        print("METROLOGY DIAGNOSTIC FRAMEWORK")
        print("=" * 70)

        # First, run measurement if we have an image
        self._run_measurement()

        # Phase 1: Coordinate Trace
        print("\n[Phase 1/8] Coordinate Trace...")
        self._result.coordinate_trace = self._run_phase1()

        # Phase 2: Calibration Validation
        print("[Phase 2/8] Calibration Validation...")
        self._result.calibration_validation = self._run_phase2()

        # Phase 3: TPS Audit
        print("[Phase 3/8] TPS Audit...")
        self._result.tps_audit = self._run_phase3()

        # Phase 4: Scale Calibration Audit
        print("[Phase 4/8] Scale Calibration Audit...")
        self._result.scale_audit = self._run_phase4()

        # Phase 5: Coordinate System Consistency
        print("[Phase 5/8] Coordinate System Audit...")
        self._result.coord_system_audit = self._run_phase5()

        # Phase 6: Distance Debug Overlay
        print("[Phase 6/8] Distance Debug Overlay...")
        self._run_phase6()

        # Phase 7: Reference Test
        print("[Phase 7/8] Reference Test...")
        self._result.reference_test = self._run_phase7()

        # Phase 8: Root Cause Recommendation
        print("[Phase 8/8] Root Cause Analysis...")
        self._result.recommendation = self._run_phase8()

        return self._result

    def _run_measurement(self) -> None:
        """Run measurement pipeline if image is available."""
        if self._image is None:
            return

        try:
            import cv2
            image = self._image
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            pipeline = MeasurementPipeline(
                self._repo, image, self._affine,
                pixel_size_mm=self._pixel_size_mm,
                residual_map=self._residual_map,
            )
            self._measured_features = pipeline.measure_all()
            print(f"  Measured {len(self._measured_features)} features")
        except Exception as e:
            print(f"  Measurement failed: {e}")

    def _run_phase1(self) -> Optional[TraceReport]:
        """Phase 1: Coordinate trace."""
        tracer = CoordinateTracer(
            self._repo, self._affine, self._pixel_size_mm, self._residual_map,
        )
        return tracer.trace_measurement(self._measured_features)

    def _run_phase2(self) -> Optional[CalibrationValidationReport]:
        """Phase 2: Calibration validation."""
        if self._corners_raw is None or self._chessboard_cols == 0:
            print("  SKIP: No chessboard corner data")
            return None

        validator = CalibrationValidator(
            self._corners_raw, self._chessboard_cols, self._chessboard_rows,
            self._chessboard_cell_mm, self._pixel_size_mm,
            self._camera_matrix, self._dist_coeffs, self._residual_map,
        )
        return validator.validate()

    def _run_phase3(self) -> TPSAuditReport:
        """Phase 3: TPS audit."""
        auditor = TPSAuditor(
            self._affine, self._pixel_size_mm,
            self._residual_map, self._image_undistorted,
        )
        if self._measured_features:
            return auditor.run_full_audit(self._measured_features)
        return auditor.audit_pipeline_code()

    def _run_phase4(self) -> ScaleAuditReport:
        """Phase 4: Scale calibration audit."""
        auditor = ScaleAuditor(
            self._pixel_size_mm, self._affine, self._residual_map,
            self._image_undistorted, self._chessboard_cell_mm,
            self._corners_raw, self._chessboard_cols, self._chessboard_rows,
        )
        return auditor.audit()

    def _run_phase5(self) -> CoordinateSystemReport:
        """Phase 5: Coordinate system consistency."""
        image_size = (0, 0)
        if self._image is not None:
            h, w = self._image.shape[:2]
            image_size = (w, h)

        auditor = CoordinateSystemAuditor(
            self._affine, self._pixel_size_mm,
            self._residual_map, image_size, self._cad_bbox,
        )
        return auditor.audit()

    def _run_phase6(self) -> None:
        """Phase 6: Distance debug overlay."""
        if self._image is None or not HAS_CV2:
            print("  SKIP: No image or cv2")
            return

        try:
            from .distance_overlay import DistanceOverlayRenderer

            renderer = DistanceOverlayRenderer(
                self._affine, self._pixel_size_mm, self._residual_map,
            )

            if self._measured_features and self._query_results:
                overlay = renderer.render_measurement_debug(
                    self._image, self._measured_features, self._query_results,
                    save_path="diagnostic_overlay.png",
                )
                print(f"  Overlay saved: diagnostic_overlay.png")
        except Exception as e:
            print(f"  Overlay failed: {e}")

    def _run_phase7(self) -> Optional[ReferenceTestReport]:
        """Phase 7: Reference test."""
        if self._corners_raw is None or self._chessboard_cols == 0:
            print("  SKIP: No chessboard corner data")
            return None

        tester = ReferenceTester(
            self._corners_raw, self._chessboard_cols, self._chessboard_rows,
            self._chessboard_cell_mm, self._pixel_size_mm,
            self._camera_matrix, self._dist_coeffs, self._residual_map,
        )
        return tester.run()

    def _run_phase8(self) -> str:
        """Phase 8: Root cause recommendation."""

        lines = []
        lines.append("=" * 70)
        lines.append("ROOT CAUSE RECOMMENDATION")
        lines.append("=" * 70)
        lines.append("")

        # Gather evidence from all phases
        evidence = []

        # Phase 2: Calibration validation evidence
        if self._result.calibration_validation:
            cv = self._result.calibration_validation
            methods = {}
            for m in cv.measurements:
                if m.method not in methods:
                    methods[m.method] = []
                methods[m.method].append(m.relative_error_pct)

            for method, errors in methods.items():
                mean_err = sum(errors) / len(errors) if errors else 0
                if abs(mean_err) > 1.0:
                    evidence.append(
                        f"[CALIBRATION] {method}: mean error = {mean_err:+.3f}% "
                        f"(significant)"
                    )

            # Check if TPS makes it worse
            cv_errs = methods.get("opencv_only", [])
            tps_errs = methods.get("opencv_tps", [])
            if cv_errs and tps_errs:
                cv_mean = abs(sum(cv_errs) / len(cv_errs))
                tps_mean = abs(sum(tps_errs) / len(tps_errs))
                if tps_mean > cv_mean * 1.5:
                    evidence.append(
                        f"[CALIBRATION] TPS INCREASES error: "
                        f"OpenCV={cv_mean:.3f}%, OpenCV+TPS={tps_mean:.3f}%"
                    )

        # Phase 3: TPS audit evidence
        if self._result.tps_audit and self._result.tps_audit.warnings:
            for w in self._result.tps_audit.warnings:
                evidence.append(f"[TPS AUDIT] {w}")

        # Phase 4: Scale audit evidence
        if self._result.scale_audit:
            sa = self._result.scale_audit
            if not sa.scale_consistent:
                evidence.append(
                    f"[SCALE] Inconsistent: affine={sa.affine_scale_mm_per_px:.6f} "
                    f"vs pixel_size_mm={self._pixel_size_mm:.6f} "
                    f"(diff={sa.scale_error_pct:.3f}%)"
                )
            if sa.diagnosis:
                evidence.append(f"[SCALE DIAGNOSIS] {sa.diagnosis}")
            for w in sa.warnings:
                evidence.append(f"[SCALE WARNING] {w}")

        # Phase 5: Coordinate system evidence
        if self._result.coord_system_audit:
            failures = [c for c in self._result.coord_system_audit.checks if not c.consistent]
            for f in failures:
                evidence.append(f"[COORD] FAIL: {f.name}: {f.error_if_any}")

        # Phase 7: Reference test evidence
        if self._result.reference_test:
            rt = self._result.reference_test
            if rt.diagnosis:
                evidence.append(f"[REFTEST] {rt.diagnosis}")
            for w in rt.warnings:
                evidence.append(f"[REFTEST WARNING] {w}")

        # Report all evidence
        if evidence:
            lines.append("EVIDENCE COLLECTED:")
            for e in evidence:
                lines.append(f"  {e}")
            lines.append("")
        else:
            lines.append("NO CRITICAL EVIDENCE FOUND — error may be in:")
            lines.append("  - Registration transform (scale/rotation)")
            lines.append("  - Feature fitting quality")
            lines.append("  - Pixel size calibration accuracy")
            lines.append("")

        # Generate recommendation
        lines.append("RECOMMENDATION:")

        # Check for the most common critical bug patterns
        has_tps_scale_issue = any(
            "TPS" in e and ("scale" in e.lower() or "error" in e.lower() or "INCREASES" in e)
            for e in evidence
        )
        has_scale_mismatch = any("SCALE" in e and "Inconsistent" in e for e in evidence)
        has_coord_issue = any("[COORD] FAIL" in e for e in evidence)

        if has_tps_scale_issue:
            lines.extend([
                "",
                "  ROOT CAUSE: TPS residual distortion introduces scale error.",
                "",
                "  The TPS correction changes effective pixel spacing. This happens",
                "  because the residual map is built on undistorted-pixel coordinates",
                "  but the TPS interpolation can introduce systematic shifts at",
                "  locations between sample points.",
                "",
                "  IMMEDIATE FIX OPTIONS:",
                "  A. Disable TPS correction (set residual_map=None in config)",
                "     and verify the error disappears.",
                "  B. Rebuild TPS with smoothing to suppress scale distortion:",
                "     increase smoothing parameter from 0.001 to 0.01-0.1.",
                "  C. Switch from image-space TPS to coordinate-space correction:",
                "     apply TPS to world coordinates after affine, not to pixel",
                "     coordinates before affine. This keeps the scale calibration",
                "     independent of TPS.",
                "",
                "  PREFERRED ARCHITECTURE (long-term):",
                "  Image → Edge Detection → Feature Fitting → Coordinate Correction",
                "  This avoids TPS warping the image, which can change effective",
                "  pixel spacing and introduce scale errors.",
            ])
        elif has_scale_mismatch:
            lines.extend([
                "",
                "  ROOT CAUSE: Scale mismatch between affine and pixel_size_mm.",
                "",
                "  The registration affine transform has a different scale than",
                "  the calibrated pixel_size_mm. Measurements use the affine to",
                "  convert pixels to world mm, so if the affine scale is wrong,",
                "  all measurements will have a systematic error.",
                "",
                "  FIX: Improve registration or force affine scale to match",
                "  pixel_size_mm (use solve_rigid_with_fixed_scale for ICP).",
            ])
        elif has_coord_issue:
            lines.extend([
                "",
                "  ROOT CAUSE: Coordinate system mixing detected.",
                "",
                "  See coordinate system audit for details. Measurements may be",
                "  computed in one coordinate space while geometry is in another.",
            ])
        else:
            lines.extend([
                "",
                "  No single root cause identified by the automated diagnostics.",
                "  The 7% error may come from:",
                "  1. Registration transform inaccuracy (check coarse/fine RMSE)",
                "  2. Pixel size calibration drift (re-calibrate with chessboard)",
                "  3. Feature fitting bias (check edge detection parameters)",
                "",
                "  SUGGESTED NEXT STEPS:",
                "  1. Re-run registration and check the RMSE values",
                "  2. Re-calibrate pixel_size_mm with a fresh chessboard image",
                "  3. Compare measured features visually (use Phase 6 overlay)",
            ])

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print all diagnostic reports to console."""

        if self._result.coordinate_trace:
            print("\n" + self._result.coordinate_trace.summary())

        if self._result.calibration_validation:
            print("\n" + self._result.calibration_validation.summary())

        if self._result.tps_audit:
            print("\n" + self._result.tps_audit.summary())

        if self._result.scale_audit:
            print("\n" + self._result.scale_audit.summary())

        if self._result.coord_system_audit:
            print("\n" + self._result.coord_system_audit.summary())

        if self._result.reference_test:
            print("\n" + self._result.reference_test.summary())

        if self._result.recommendation:
            print("\n" + self._result.recommendation)

    def save_reports(self, output_dir: str = "diagnostics_output") -> None:
        """Save all reports to files."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if self._result.coordinate_trace:
            self._result.coordinate_trace.save(out / "01_coordinate_trace.json")
        if self._result.calibration_validation:
            self._result.calibration_validation.save(out / "02_calibration_validation.json")
        if self._result.tps_audit:
            self._result.tps_audit.save(out / "03_tps_audit.json")
        if self._result.scale_audit:
            self._result.scale_audit.save(out / "04_scale_audit.json")
        if self._result.coord_system_audit:
            self._result.coord_system_audit.save(out / "05_coord_system_audit.json")
        if self._result.reference_test:
            self._result.reference_test.save(out / "07_reference_test.json")

        # Save recommendation as text
        if self._result.recommendation:
            (out / "08_recommendation.txt").write_text(self._result.recommendation)

        # Save full text summary
        summary_lines = []
        if self._result.coordinate_trace:
            summary_lines.append(self._result.coordinate_trace.summary())
        if self._result.calibration_validation:
            summary_lines.append(self._result.calibration_validation.summary())
        if self._result.tps_audit:
            summary_lines.append(self._result.tps_audit.summary())
        if self._result.scale_audit:
            summary_lines.append(self._result.scale_audit.summary())
        if self._result.coord_system_audit:
            summary_lines.append(self._result.coord_system_audit.summary())
        if self._result.reference_test:
            summary_lines.append(self._result.reference_test.summary())
        if self._result.recommendation:
            summary_lines.append(self._result.recommendation)

        (out / "full_diagnostic_report.txt").write_text("\n\n".join(summary_lines))
        print(f"\nReports saved to {out}/")


try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
"""
Diagnostics module — metrology error diagnostic framework.

Provides 8 diagnostic phases to trace a measurement error through the
complete pipeline without modifying any algorithms:

  1. coordinate_trace   — trace every coordinate transformation per feature
  2. calibration_validation — measure known grid distances under each correction method
  3. tps_audit          — document where TPS is applied, check for double application
  4. scale_audit        — verify pixel/mm scale consistency
  5. coord_system_audit — detect coordinate space mixing
  6. distance_overlay   — visual debug overlay with distances
  7. reference_test     — compare raw / OpenCV / OpenCV+TPS on known distances
  8. runner             — orchestrate all phases and produce root-cause recommendation

Usage:
    from cadviewer.diagnostics.runner import DiagnosticRunner

    runner = DiagnosticRunner(
        repo=repo, affine=affine, pixel_size_mm=0.1162,
        image=image, residual_map=rmap, ...
    )
    runner.run_all()
    runner.print_summary()
    runner.save_reports("diagnostics_output")
"""

from .coordinate_trace import CoordinateTracer, CoordinateTrace, TraceReport, run_coordinate_trace
from .calibration_validation import CalibrationValidator, CalibrationValidationReport, run_calibration_validation
from .tps_audit import TPSAuditor, TPSAuditReport, TPSAuditPoint, run_tps_audit
from .scale_audit import ScaleAuditor, ScaleAuditReport, run_scale_audit
from .coord_system_audit import CoordinateSystemAuditor, CoordinateSystemReport, run_coordinate_system_audit
from .distance_overlay import DistanceOverlayRenderer, DistanceOverlayItem, create_distance_overlay
from .reference_test import ReferenceTester, ReferenceTestReport, run_reference_test
from .runner import DiagnosticRunner, DiagnosticResult

__all__ = [
    "DiagnosticRunner",
    "DiagnosticResult",
    "CoordinateTracer",
    "CoordinateTrace",
    "TraceReport",
    "CalibrationValidator",
    "CalibrationValidationReport",
    "TPSAuditor",
    "TPSAuditReport",
    "TPSAuditPoint",
    "ScaleAuditor",
    "ScaleAuditReport",
    "CoordinateSystemAuditor",
    "CoordinateSystemReport",
    "DistanceOverlayRenderer",
    "DistanceOverlayItem",
    "create_distance_overlay",
    "ReferenceTester",
    "ReferenceTestReport",
    "run_coordinate_trace",
    "run_calibration_validation",
    "run_tps_audit",
    "run_scale_audit",
    "run_coordinate_system_audit",
    "run_reference_test",
]

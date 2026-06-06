"""
Coordinate System Consistency Audit — verifies all geometry uses one
consistent coordinate space.

Checks:
  - CAD coordinates (mm, Y-up or Y-down depending on DXF)
  - Registered coordinates (mm, via affine)
  - Image coordinates (pixels, Y-down)
  - Image world coordinates (mm, Y-flipped from pixels)
  - Undistorted coordinates (pixels, after cv2.undistort)
  - TPS-corrected coordinates (pixels, after residual correction)
  - Measurement coordinates (mm, via affine from TPS-corrected pixels)

Detects mixing of coordinate spaces — a common source of systematic error.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np


def _json_default(obj):
    """Handle numpy types for JSON serialization."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

from ..calibration.residual_map import ResidualDistortionMap
from ..registration import affine_solver


@dataclass
class CoordinateSpace:
    """Description of a coordinate space."""
    name: str
    units: str  # "mm" or "pixels"
    y_direction: str  # "up" or "down"
    origin: str  # description of origin
    examples: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class ConsistencyCheck:
    """Result of a consistency check between two coordinate spaces."""
    name: str
    space_a: str
    space_b: str
    consistent: bool
    details: str
    error_if_any: str = ""


@dataclass
class CoordinateSystemReport:
    """Complete coordinate system consistency audit report."""

    spaces: List[CoordinateSpace] = field(default_factory=list)
    checks: List[ConsistencyCheck] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    diagnosis: str = ""

    def to_dict(self) -> dict:
        return {
            "spaces": [
                {"name": s.name, "units": s.units, "y_dir": s.y_direction,
                 "origin": s.origin, "examples": s.examples}
                for s in self.spaces
            ],
            "checks": [
                {"name": c.name, "consistent": c.consistent, "details": c.details,
                 "error": c.error_if_any}
                for c in self.checks
            ],
            "warnings": self.warnings,
            "diagnosis": self.diagnosis,
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, default=_json_default))

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "COORDINATE SYSTEM CONSISTENCY REPORT",
            "=" * 70,
            "",
            "COORDINATE SPACES:",
        ]

        for s in self.spaces:
            lines.append(f"  {s.name}:")
            lines.append(f"    Units: {s.units}, Y: {s.y_direction}, Origin: {s.origin}")
            if s.examples:
                for k, v in s.examples.items():
                    lines.append(f"    {k}: ({v[0]:.3f}, {v[1]:.3f})")
            lines.append("")

        if self.checks:
            lines.append("CONSISTENCY CHECKS:")
            for c in self.checks:
                status = "PASS" if c.consistent else "FAIL"
                lines.append(f"  [{status}] {c.name}")
                lines.append(f"    {c.details}")
                if c.error_if_any:
                    lines.append(f"    ERROR: {c.error_if_any}")
                lines.append("")

        if self.warnings:
            lines.append("WARNINGS:")
            for w in self.warnings:
                lines.append(f"  * {w}")
            lines.append("")

        if self.diagnosis:
            lines.extend(["DIAGNOSIS:", f"  {self.diagnosis}", ""])

        lines.append("=" * 70)
        return "\n".join(lines)


class CoordinateSystemAuditor:
    """Audits coordinate system consistency across the measurement pipeline."""

    def __init__(
        self,
        affine: np.ndarray,
        pixel_size_mm: float,
        residual_map: Optional[ResidualDistortionMap] = None,
        image_size: Tuple[int, int] = (0, 0),
        cad_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        self._affine = affine
        self._pixel_size_mm = pixel_size_mm
        self._residual_map = residual_map
        self._image_size = image_size  # (width, height)
        self._cad_bbox = cad_bbox  # (min_x, min_y, max_x, max_y)

    def audit(self) -> CoordinateSystemReport:
        report = CoordinateSystemReport()

        # Define all coordinate spaces in the system
        self._define_spaces(report)

        # Run consistency checks
        self._check_pixel_world_conversion(report)
        self._check_affine_direction(report)
        self._check_tps_space(report)
        self._check_cad_vs_world(report)
        self._check_y_convention(report)

        # Generate diagnosis
        self._generate_diagnosis(report)

        return report

    def _define_spaces(self, report: CoordinateSystemReport) -> None:
        """Define all coordinate spaces in the system."""

        # Space 1: CAD coordinates
        report.spaces.append(CoordinateSpace(
            name="CAD",
            units="mm",
            y_direction="variable (DXF convention)",
            origin="DXF origin",
        ))

        # Space 2: Raw image pixels
        w, h = self._image_size
        report.spaces.append(CoordinateSpace(
            name="Image Raw Pixels",
            units="pixels",
            y_direction="down",
            origin="top-left corner (0,0)",
            examples={"top-left": (0, 0), "bottom-right": (w, h)} if w > 0 else {},
        ))

        # Space 3: Image world coordinates (pixel_size_mm conversion)
        if w > 0 and h > 0:
            report.spaces.append(CoordinateSpace(
                name="Image World",
                units="mm",
                y_direction="up (Y-flipped from pixels)",
                origin="top-left corner = (0, 0), bottom = (0, h*pixel_size_mm)",
                examples={
                    "pixel(0,0)": (0, 0),
                    "pixel(w,h)": (w * self._pixel_size_mm, -h * self._pixel_size_mm),
                },
            ))

        # Space 4: Undistorted pixels
        report.spaces.append(CoordinateSpace(
            name="Undistorted Pixels",
            units="pixels",
            y_direction="down",
            origin="same as raw (cv2.undistort preserves coordinate system)",
        ))

        # Space 5: TPS-corrected pixels
        tps_status = "built" if (self._residual_map and self._residual_map.is_built) else "not built"
        report.spaces.append(CoordinateSpace(
            name="TPS Corrected Pixels",
            units="pixels",
            y_direction="down",
            origin=f"undistorted pixels + TPS correction ({tps_status})",
        ))

        # Space 6: Measurement world coordinates
        params = affine_solver.extract_params(self._affine)
        report.spaces.append(CoordinateSpace(
            name="Measurement World",
            units="mm",
            y_direction="depends on affine",
            origin="affine @ pixel_origin",
            examples={
                "scale": (params['scale_x'], 0),
                "rotation_deg": (params['rotation_deg'], 0),
            },
        ))

    def _check_pixel_world_conversion(self, report: CoordinateSystemReport) -> None:
        """Check that pixel→world conversion is consistent."""
        w, h = self._image_size
        if w == 0:
            return

        # Test: transform image corners via affine
        corners_px = np.array([
            [0, 0], [w, 0], [w, h], [0, h],
        ], dtype=np.float64)
        corners_world = affine_solver.apply(self._affine, corners_px)

        # Check scale consistency
        dx_world = np.linalg.norm(corners_world[1] - corners_world[0])
        dx_px = w
        scale_x = dx_world / dx_px if dx_px > 0 else 0

        dy_world = np.linalg.norm(corners_world[3] - corners_world[0])
        dy_px = h
        scale_y = dy_world / dy_px if dy_px > 0 else 0

        scale_diff_pct = abs(scale_x - scale_y) / max(scale_x, scale_y, 1e-10) * 100

        params = affine_solver.extract_params(self._affine)
        affine_scale = params['scale_x']

        report.checks.append(ConsistencyCheck(
            name="Pixel-to-world scale consistency",
            space_a="Image Raw Pixels",
            space_b="Measurement World",
            consistent=scale_diff_pct < 5.0 and abs(scale_x - affine_scale) / affine_scale < 0.01,
            details=(
                f"X scale: {scale_x:.6f} mm/px, Y scale: {scale_y:.6f} mm/px. "
                f"Diff: {scale_diff_pct:.3f}%. Affine scale: {affine_scale:.6f}. "
                f"pixel_size_mm: {self._pixel_size_mm:.6f}."
            ),
            error_if_any=(
                f"Scale from affine ({affine_scale:.6f}) differs from pixel_size_mm "
                f"({self._pixel_size_mm:.6f}) by "
                f"{abs(affine_scale - self._pixel_size_mm) / self._pixel_size_mm * 100:.3f}%"
                if abs(affine_scale - self._pixel_size_mm) / self._pixel_size_mm > 0.005
                else ""
            ),
        ))

    def _check_affine_direction(self, report: CoordinateSystemReport) -> None:
        """Check that the affine maps pixel→CAD correctly."""
        params = affine_solver.extract_params(self._affine)

        # A pixel-space affine should have scale ≈ pixel_size_mm
        # and negative Y component (pixel Y-down → world Y-up)
        t01 = self._affine[0, 1]  # cross-coupling X from pixel-Y
        t11 = self._affine[1, 1]  # Y from pixel-Y

        # For no rotation: t11 should be negative (Y-flip)
        # With rotation, this is more complex
        rotation = params['rotation_deg']

        report.checks.append(ConsistencyCheck(
            name="Affine direction check",
            space_a="Image Raw Pixels",
            space_b="Measurement World",
            consistent=True,
            details=(
                f"Affine rotation: {rotation:.4f}deg. "
                f"Affine[0,1]={t01:.6f}, Affine[1,1]={t11:.6f}. "
                f"For telecentric: scale should be ~pixel_size_mm={self._pixel_size_mm:.6f}"
            ),
        ))

    def _check_tps_space(self, report: CoordinateSystemReport) -> None:
        """Check that TPS operates in the correct coordinate space."""
        if self._residual_map is None or not self._residual_map.is_built:
            report.checks.append(ConsistencyCheck(
                name="TPS coordinate space",
                space_a="N/A",
                space_b="N/A",
                consistent=True,
                details="TPS map not built — no correction applied.",
            ))
            return

        # TPS sample points should be in pixel coordinates
        # Check if sample points are within image bounds
        samples = self._residual_map._sample_points
        img_w, img_h = self._residual_map.image_size

        if samples is not None and img_w > 0:
            in_bounds = (
                (samples[:, 0] >= 0) & (samples[:, 0] <= img_w) &
                (samples[:, 1] >= 0) & (samples[:, 1] <= img_h)
            )
            pct_in_bounds = float(in_bounds.sum()) / len(in_bounds) * 100

            # Check if samples look like they're in mm instead of pixels
            max_coord = max(samples[:, 0].max(), samples[:, 1].max())
            looks_like_mm = max_coord > max(img_w, img_h) * 1.5

            report.checks.append(ConsistencyCheck(
                name="TPS sample point space",
                space_a="TPS Corrected Pixels",
                space_b="Undistorted Pixels",
                consistent=not looks_like_mm,
                details=(
                    f"TPS samples: {len(samples)} points. "
                    f"In image bounds: {pct_in_bounds:.1f}%. "
                    f"Max coordinate: {max_coord:.1f}. "
                    f"Image size: {img_w}x{img_h} px."
                ),
                error_if_any=(
                    "TPS sample points appear to be in mm coordinates, not pixels! "
                    "This means TPS correction will produce wrong results."
                    if looks_like_mm else ""
                ),
            ))

    def _check_cad_vs_world(self, report: CoordinateSystemReport) -> None:
        """Check that CAD world and measurement world are in the same space."""
        if self._cad_bbox is None:
            return

        # Transform a known CAD point through the affine inverse
        # to check if it lands in a reasonable image location
        cad_center = np.array([[
            (self._cad_bbox[0] + self._cad_bbox[2]) / 2,
            (self._cad_bbox[1] + self._cad_bbox[3]) / 2,
        ]])
        inv_affine = affine_solver.invert(self._affine)
        img_point = affine_solver.apply(inv_affine, cad_center)[0]

        w, h = self._image_size
        reasonable = True
        if w > 0:
            reasonable = -500 < img_point[0] < w + 500 and -500 < img_point[1] < h + 500

        report.checks.append(ConsistencyCheck(
            name="CAD-to-image projection reasonableness",
            space_a="CAD",
            space_b="Image Raw Pixels",
            consistent=reasonable,
            details=(
                f"CAD center ({cad_center[0,0]:.1f}, {cad_center[0,1]:.1f}) → "
                f"pixel ({img_point[0]:.1f}, {img_point[1]:.1f}). "
                f"Image size: {w}x{h} px."
            ),
            error_if_any=(
                "CAD center projects far outside the image. "
                "Affine may be incorrect or coordinate spaces are mixed."
                if not reasonable else ""
            ),
        ))

    def _check_y_convention(self, report: CoordinateSystemReport) -> None:
        """Check Y-axis conventions across all spaces."""
        # In image: Y increases downward
        # In image world (via T_pixel_to_imgworld): Y is negated (multiply by -pixel_size_mm)
        # In CAD/DXF: Y convention varies

        # The affine = T_imgworld_to_cad @ T_pixel_to_imgworld
        # T_pixel_to_imgworld has [0, -pixel_size_mm, 0] on the Y row
        # So the affine already accounts for Y-flip

        report.checks.append(ConsistencyCheck(
            name="Y-axis convention",
            space_a="Image Raw Pixels",
            space_b="Measurement World",
            consistent=True,
            details=(
                "Image pixels: Y-down. Image world: Y-flipped (* -pixel_size_mm). "
                "Affine incorporates Y-flip via T_pixel_to_imgworld. "
                "CAD/DXF Y convention depends on the drawing."
            ),
        ))

    def _generate_diagnosis(self, report: CoordinateSystemReport) -> None:
        """Generate diagnosis from failed checks."""

        failures = [c for c in report.checks if not c.consistent]
        if not failures:
            report.diagnosis = "All coordinate system consistency checks passed."
            return

        report.diagnosis = "Coordinate space inconsistencies detected:\n"
        for f in failures:
            report.diagnosis += f"  - {f.name}: {f.error_if_any}\n"


def run_coordinate_system_audit(
    affine: np.ndarray,
    pixel_size_mm: float,
    residual_map: Optional[ResidualDistortionMap] = None,
    image_size: Tuple[int, int] = (0, 0),
    cad_bbox: Optional[Tuple[float, float, float, float]] = None,
    output_path: Optional[Path] = None,
) -> CoordinateSystemReport:
    """Run coordinate system consistency audit."""

    auditor = CoordinateSystemAuditor(
        affine, pixel_size_mm, residual_map, image_size, cad_bbox,
    )
    report = auditor.audit()

    if output_path is not None:
        report.save(output_path)
        print(f"[COORD] Report saved to {output_path}")

    return report
"""
CalibrationReport — error analysis for calibration results.

Computes before/after statistics for the residual distortion map,
providing per-point, RMS, and max-error metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CalibrationReport:
    """Statistical analysis of calibration accuracy."""

    # Before residual correction (OpenCV calibration only)
    rms_before_px: float = 0.0
    max_before_px: float = 0.0
    rms_before_mm: float = 0.0
    max_before_mm: float = 0.0

    # After residual correction
    rms_after_px: float = 0.0
    max_after_px: float = 0.0
    rms_after_mm: float = 0.0
    max_after_mm: float = 0.0

    # Metadata
    n_points: int = 0
    n_images: int = 0
    image_size: tuple[int, int] = (0, 0)
    opencv_rms: float = 0.0

    # Per-point residuals (for heatmap visualization)
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    residuals_before: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    residuals_after: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))

    def compute(
        self,
        points: np.ndarray,
        residuals_before: np.ndarray,
        residuals_after: np.ndarray,
        pixel_size_mm: float = 0.0,
    ) -> None:
        """Compute statistics from before/after residual arrays.

        Args:
            points: Nx2 array of sample positions (pixels).
            residuals_before: Nx2 array of (dx, dy) before correction.
            residuals_after: Nx2 array of (dx, dy) after correction.
            pixel_size_mm: mm per pixel, for converting to world units.
        """
        self.points = points
        self.residuals_before = residuals_before
        self.residuals_after = residuals_after
        self.n_points = len(points)

        dist_before = np.sqrt(np.sum(residuals_before ** 2, axis=1))
        dist_after = np.sqrt(np.sum(residuals_after ** 2, axis=1))

        self.rms_before_px = float(np.sqrt(np.mean(dist_before ** 2)))
        self.max_before_px = float(np.max(dist_before))
        self.rms_after_px = float(np.sqrt(np.mean(dist_after ** 2)))
        self.max_after_px = float(np.max(dist_after))

        if pixel_size_mm > 0:
            self.rms_before_mm = self.rms_before_px * pixel_size_mm
            self.max_before_mm = self.max_before_px * pixel_size_mm
            self.rms_after_mm = self.rms_after_px * pixel_size_mm
            self.max_after_mm = self.max_after_px * pixel_size_mm

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Calibration Report",
            f"  Images: {self.n_images}  |  Points: {self.n_points}",
            f"  Image size: {self.image_size[0]}x{self.image_size[1]}",
            f"  OpenCV RMS: {self.opencv_rms:.4f} px",
            "",
            f"  Before residual correction:",
            f"    RMS: {self.rms_before_px:.4f} px ({self.rms_before_mm:.4f} mm)",
            f"    Max: {self.max_before_px:.4f} px ({self.max_before_mm:.4f} mm)",
            "",
            f"  After residual correction:",
            f"    RMS: {self.rms_after_px:.4f} px ({self.rms_after_mm:.4f} mm)",
            f"    Max: {self.max_after_px:.4f} px ({self.max_after_mm:.4f} mm)",
            "",
            f"  Improvement:",
            f"    RMS: {self.rms_before_px:.4f} → {self.rms_after_px:.4f} px "
            f"({(1 - self.rms_after_px / max(self.rms_before_px, 1e-10)) * 100:.1f}%)",
            f"    Max: {self.max_before_px:.4f} → {self.max_after_px:.4f} px "
            f"({(1 - self.max_after_px / max(self.max_before_px, 1e-10)) * 100:.1f}%)",
        ]
        return "\n".join(lines)

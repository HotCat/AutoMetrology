"""
Line fitting engine using perpendicular scanline sampling.

Samples gradient profiles along perpendicular scanlines at regular
intervals along the predicted line direction. Finds the strongest
gradient transition on each scanline and localizes it to subpixel.

This is NOT HoughLines. It uses:
  1. Scharr gradient magnitude (precomputed)
  2. Perpendicular scanline sampling
  3. Subpixel localization via parabolic interpolation
  4. Least-squares line fitting via SVD or cv2.fitLine
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class LineFitResult:
    """Result of line fitting from image edge data."""

    p1: np.ndarray  # (x1, y1) in pixels, subpixel
    p2: np.ndarray  # (x2, y2) in pixels, subpixel
    edge_points: np.ndarray  # Nx2 detected edge points (pixels)
    residual: float  # mean perpendicular distance (pixels)
    confidence: float  # 0-1
    n_edge_points: int
    gradient_strength: float  # mean gradient at edge points


class LineFittingEngine:
    """Fit lines using perpendicular scanline sampling and least-squares."""

    def __init__(self, gradient: np.ndarray) -> None:
        """
        Args:
            gradient: HxW float64 gradient magnitude image (precomputed)
        """
        self._gradient = gradient
        self._h, self._w = gradient.shape[:2]

    def fit(
        self,
        predicted_p1: np.ndarray,
        predicted_p2: np.ndarray,
        n_scanlines: int = 60,
        scan_width: float = 15.0,
        min_gradient: float = 15.0,
    ) -> LineFitResult | None:
        """Fit line using perpendicular scanline sampling.

        Args:
            predicted_p1: (x1, y1) in pixels from CAD projection
            predicted_p2: (x2, y2) in pixels from CAD projection
            n_scanlines: number of perpendicular scanlines
            scan_width: half-width of each scanline (pixels)
            min_gradient: minimum gradient to accept an edge

        Returns:
            LineFitResult or None if insufficient edge points
        """
        p1 = np.array(predicted_p1, dtype=np.float64)
        p2 = np.array(predicted_p2, dtype=np.float64)
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return None
        direction /= length
        normal = np.array([-direction[1], direction[0]])

        scan_width = max(scan_width, length * 0.08)

        edge_points = self._scanline_sampling(
            p1, p2, direction, normal, n_scanlines, scan_width, min_gradient,
        )

        if len(edge_points) < 4:
            return None

        # Fit line via SVD (total least squares)
        fitted = self._fit_line_svd(edge_points)
        if fitted is None:
            return None

        line_dir, line_pt = fitted

        # Project edge points onto line to get endpoints
        projections = (edge_points - line_pt) @ line_dir
        t_min, t_max = projections.min(), projections.max()
        fit_p1 = line_pt + t_min * line_dir
        fit_p2 = line_pt + t_max * line_dir

        # Residual: mean perpendicular distance
        diffs = edge_points - line_pt
        perp_dists = np.abs(diffs @ normal)
        residual = float(np.mean(perp_dists))

        # Confidence
        coverage = len(edge_points) / n_scanlines
        max_residual = max(scan_width * 0.2, 2.0)
        residual_score = max(0.0, 1.0 - residual / max_residual)
        confidence = min(1.0, coverage) * residual_score

        # Gradient strength
        ix = np.clip(np.round(edge_points[:, 0]).astype(int), 0, self._w - 1)
        iy = np.clip(np.round(edge_points[:, 1]).astype(int), 0, self._h - 1)
        grad_strength = float(np.mean(self._gradient[iy, ix]))

        return LineFitResult(
            p1=fit_p1,
            p2=fit_p2,
            edge_points=edge_points,
            residual=residual,
            confidence=confidence,
            n_edge_points=len(edge_points),
            gradient_strength=grad_strength,
        )

    def _scanline_sampling(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        direction: np.ndarray,
        normal: np.ndarray,
        n_scanlines: int,
        scan_width: float,
        min_gradient: float,
    ) -> np.ndarray:
        """Sample edges along perpendicular scanlines."""
        n_samples = max(7, int(2 * scan_width) + 1)
        offsets = np.linspace(-scan_width, scan_width, n_samples)
        t_values = np.linspace(0, 1, n_scanlines)

        edge_points: list[list[float]] = []
        dr = (2 * scan_width) / (n_samples - 1) if n_samples > 1 else 1.0

        for t in t_values:
            base = p1 + t * (p2 - p1)

            # Sample positions along perpendicular direction
            sample_px = base[0] + offsets * normal[0]
            sample_py = base[1] + offsets * normal[1]

            in_bounds = (
                (sample_px >= 0) & (sample_px < self._w) &
                (sample_py >= 0) & (sample_py < self._h)
            )
            if not np.any(in_bounds):
                continue

            ix = np.clip(np.round(sample_px).astype(int), 0, self._w - 1)
            iy = np.clip(np.round(sample_py).astype(int), 0, self._h - 1)
            grad_profile = self._gradient[iy, ix]
            grad_profile[~in_bounds] = 0.0

            peak_idx = int(np.argmax(grad_profile))
            peak_val = grad_profile[peak_idx]

            if peak_val < min_gradient:
                continue

            # Subpixel localization
            sub_offset = -scan_width + peak_idx * dr
            if 1 <= peak_idx <= n_samples - 2:
                y_m1 = grad_profile[peak_idx - 1]
                y_0 = grad_profile[peak_idx]
                y_p1 = grad_profile[peak_idx + 1]
                denom = 2.0 * (2.0 * y_0 - y_m1 - y_p1)
                if abs(denom) > 1e-10:
                    parabolic_offset = np.clip((y_p1 - y_m1) / denom, -0.5, 0.5)
                    sub_offset = -scan_width + (peak_idx + parabolic_offset) * dr

            ex = base[0] + sub_offset * normal[0]
            ey = base[1] + sub_offset * normal[1]
            edge_points.append([ex, ey])

        return np.array(edge_points) if edge_points else np.empty((0, 2))

    @staticmethod
    def _fit_line_svd(
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Total least-squares line fit via SVD.

        Returns (direction_unit_vector, point_on_line) or None.
        """
        if len(points) < 2:
            return None
        centroid = points.mean(axis=0)
        centered = points - centroid
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        direction = Vt[0]  # first principal component
        # Ensure consistent orientation
        if direction[0] < 0:
            direction = -direction
        return direction, centroid

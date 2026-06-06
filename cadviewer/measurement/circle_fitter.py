"""
Circle fitting engine using radial edge sampling.

Industrial-style approach: cast radial rays from the predicted center,
detect the strongest gradient transition along each ray near the
predicted radius, and fit a circle to the collected edge points.

This is NOT HoughCircles. It uses:
  1. Scharr gradient magnitude (precomputed)
  2. Radial ray sampling from predicted center
  3. Subpixel localization via parabolic interpolation
  4. Least-squares circle fitting (Kasa algebraic method)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CircleFitResult:
    """Result of circle fitting from image edge data."""

    center: np.ndarray  # (cx, cy) in pixels, subpixel
    radius: float  # pixels
    edge_points: np.ndarray  # Nx2 detected edge points (pixels)
    residual: float  # mean radial residual (pixels)
    confidence: float  # 0-1
    n_edge_points: int
    gradient_strength: float  # mean gradient at edge points


class CircleFittingEngine:
    """Fit circles using radial edge sampling and least-squares fitting."""

    def __init__(self, gradient: np.ndarray) -> None:
        """
        Args:
            gradient: HxW float64 gradient magnitude image (precomputed)
        """
        self._gradient = gradient
        self._h, self._w = gradient.shape[:2]

    def fit(
        self,
        predicted_center: np.ndarray,
        predicted_radius: float,
        n_rays: int = 90,
        search_width_ratio: float = 0.25,
        min_gradient: float = 15.0,
    ) -> CircleFitResult | None:
        """Fit circle using radial edge sampling.

        Args:
            predicted_center: (cx, cy) in pixels from CAD projection
            predicted_radius: expected radius in pixels from CAD projection
            n_rays: number of radial rays to cast
            search_width_ratio: search band width as fraction of radius
            min_gradient: minimum gradient magnitude to accept an edge

        Returns:
            CircleFitResult or None if insufficient edge points
        """
        search_width = max(predicted_radius * search_width_ratio, 5.0)
        edge_points = self._radial_edge_sampling(
            predicted_center, predicted_radius, n_rays, search_width, min_gradient,
        )

        if len(edge_points) < 8:
            return None

        # Least-squares circle fit (Kasa algebraic method)
        fitted = self._fit_circle_kasa(edge_points)
        if fitted is None:
            return None

        cx, cy, r = fitted
        center = np.array([cx, cy])

        # Residual
        dists = np.sqrt((edge_points[:, 0] - cx) ** 2 + (edge_points[:, 1] - cy) ** 2)
        residual = float(np.mean(np.abs(dists - r)))

        # Confidence: based on coverage and residual
        coverage = len(edge_points) / n_rays
        max_residual = max(predicted_radius * 0.1, 2.0)
        residual_score = max(0.0, 1.0 - residual / max_residual)
        confidence = min(1.0, coverage) * residual_score

        # Mean gradient strength at edge points
        ix = np.clip(np.round(edge_points[:, 0]).astype(int), 0, self._w - 1)
        iy = np.clip(np.round(edge_points[:, 1]).astype(int), 0, self._h - 1)
        grad_strength = float(np.mean(self._gradient[iy, ix]))

        # Reject fits where edge points have low gradient (noise, not real edges)
        image_grad_mean = float(np.mean(self._gradient))
        if grad_strength < max(min_gradient * 3.0, image_grad_mean * 3.0):
            return None

        return CircleFitResult(
            center=center,
            radius=r,
            edge_points=edge_points,
            residual=residual,
            confidence=confidence,
            n_edge_points=len(edge_points),
            gradient_strength=grad_strength,
        )

    def _radial_edge_sampling(
        self,
        center: np.ndarray,
        radius: float,
        n_rays: int,
        search_width: float,
        min_gradient: float,
    ) -> np.ndarray:
        """Sample edges along radial rays from center.

        For each ray, scans from (radius - search_width) to (radius + search_width),
        finds the gradient peak closest to the predicted radius, and localizes it
        to subpixel via parabolic interpolation.

        Uses closest-to-predicted strategy: among multiple edges (chamfer, shadow),
        the one nearest the CAD-predicted radius is the true feature boundary.
        """
        cx, cy = float(center[0]), float(center[1])
        r_start = max(1.0, radius - search_width)
        r_end = radius + search_width
        n_samples = max(7, int(2 * search_width) + 1)

        angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
        sample_radii = np.linspace(r_start, r_end, n_samples)

        cos_a = np.cos(angles)[:, None]
        sin_a = np.sin(angles)[:, None]
        r = sample_radii[None, :]

        px = cx + r * cos_a
        py = cy + r * sin_a

        in_bounds = (px >= 0) & (px < self._w) & (py >= 0) & (py < self._h)

        ix = np.clip(np.round(px).astype(int), 0, self._w - 1)
        iy = np.clip(np.round(py).astype(int), 0, self._h - 1)
        grad_profile = self._gradient[iy, ix]
        grad_profile[~in_bounds] = 0.0

        dr = (r_end - r_start) / (n_samples - 1) if n_samples > 1 else 1.0
        # Center index = predicted radius position
        center_idx = n_samples // 2

        edge_points: list[list[float]] = []

        for i in range(n_rays):
            profile = grad_profile[i]
            pi = self._find_closest_peak(profile, min_gradient, center_idx)
            if pi is None:
                continue

            sub_r = r_start + pi * dr

            # Parabolic subpixel refinement
            if 1 <= pi <= n_samples - 2:
                y_m1 = profile[pi - 1]
                y_0 = profile[pi]
                y_p1 = profile[pi + 1]
                denom = 2.0 * (2.0 * y_0 - y_m1 - y_p1)
                if abs(denom) > 1e-10:
                    offset = np.clip((y_p1 - y_m1) / denom, -0.5, 0.5)
                    sub_r = r_start + (pi + offset) * dr

            ex = cx + sub_r * cos_a[i, 0]
            ey = cy + sub_r * sin_a[i, 0]
            edge_points.append([ex, ey])

        return np.array(edge_points) if edge_points else np.empty((0, 2))

    @staticmethod
    def _find_closest_peak(
        profile: np.ndarray, min_gradient: float, center: int,
    ) -> int | None:
        """Find the gradient peak closest to center index.

        Requires peaks to have PROMINENCE: the peak value must be at least
        2x the profile mean. This rejects noise/texture and only accepts
        real edge transitions. Falls back to any sample above the adaptive
        threshold if no prominent peak found.
        """
        n = len(profile)
        # Adaptive threshold: must be above both min_gradient AND the
        # profile background. This prevents the fitter from locking onto
        # texture noise when the ROI misses the real feature edge.
        # Use 3x the profile mean to ensure we only accept strong edges.
        profile_mean = float(np.mean(profile))
        adaptive_threshold = max(min_gradient, profile_mean * 3.0)

        # Find all local maxima above adaptive threshold
        peaks = []
        for j in range(1, n - 1):
            if (profile[j] >= adaptive_threshold
                    and profile[j] >= profile[j - 1]
                    and profile[j] >= profile[j + 1]):
                peaks.append(j)
        # Check endpoints
        if n > 0 and profile[0] >= adaptive_threshold and profile[0] >= profile[min(1, n - 1)]:
            peaks.append(0)
        if n > 1 and profile[-1] >= adaptive_threshold and profile[-1] >= profile[-2]:
            peaks.append(n - 1)

        if peaks:
            return min(peaks, key=lambda j: abs(j - center))

        # Fallback: any sample above adaptive threshold, closest to center
        above = [j for j in range(n) if profile[j] >= adaptive_threshold]
        if above:
            return min(above, key=lambda j: abs(j - center))
        return None

    @staticmethod
    def _fit_circle_kasa(points: np.ndarray) -> tuple[float, float, float] | None:
        """Kasa algebraic circle fit via least squares.

        Solves: [x, y, 1] @ [2*cx, 2*cy, r^2-cx^2-cy^2] = x^2 + y^2
        """
        if len(points) < 3:
            return None
        x = points[:, 0]
        y = points[:, 1]
        A = np.column_stack([x, y, np.ones(len(x))])
        b = x ** 2 + y ** 2
        try:
            result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
        cx = result[0] / 2.0
        cy = result[1] / 2.0
        r_sq = result[2] + cx ** 2 + cy ** 2
        if r_sq < 0:
            return None
        return cx, cy, math.sqrt(r_sq)

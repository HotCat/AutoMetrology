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
        preferred_side_point: np.ndarray | None = None,
        max_scan_width: float | None = None,
        prefer_extreme_side: bool = False,
        lock_direction: bool = False,
    ) -> LineFitResult | None:
        """Fit line using iterative perpendicular scanline sampling.

        Runs up to 3 iterations: after each SVD fit, re-scans perpendicular
        to the fitted direction. This eliminates systematic bias caused by
        scanlines not being perpendicular to the actual edge.

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
        reference_direction = direction.copy()

        scan_width = max(scan_width, length * 0.08)
        if max_scan_width is not None:
            scan_width = min(scan_width, max(7.0, float(max_scan_width)))
        side_point = (
            np.array(preferred_side_point, dtype=np.float64)
            if preferred_side_point is not None else None
        )

        # Iterative refinement: re-scan perpendicular to fitted line
        best_edge_points = None
        for _ in range(3):
            normal = np.array([-direction[1], direction[0]])
            edge_points = self._scanline_sampling(
                p1, p2, direction, normal, n_scanlines, scan_width, min_gradient,
                side_point, prefer_extreme_side,
            )
            if len(edge_points) < 4:
                break
            clustered = self._select_offset_cluster(
                edge_points, p1, direction, side_point, prefer_extreme_side,
            )
            if len(clustered) >= 4:
                edge_points = clustered
            fitted = self._fit_line_svd(edge_points)
            if fitted is None:
                break
            new_direction, _ = fitted
            # Keep local line fits close to the CAD-projected direction. A
            # large SVD rotation means scanlines mixed multiple parallel edges
            # or end caps, not that the physical feature rotated that much.
            dot = abs(np.dot(new_direction, direction))
            best_edge_points = edge_points
            if not lock_direction and dot > 0.996195:  # cos(5 deg)
                direction = new_direction
            if dot > 0.999999:
                break

        if best_edge_points is None or len(best_edge_points) < 4:
            return None

        # Final fit
        fitted = self._fit_line_svd(best_edge_points)
        if fitted is None:
            return None

        line_dir, line_pt = fitted
        if lock_direction:
            line_dir, line_pt = self._fit_line_with_direction(
                best_edge_points, reference_direction,
            )
        elif abs(float(np.dot(line_dir, direction))) <= 0.996195:
            line_dir, line_pt = self._fit_line_with_direction(best_edge_points, direction)

        # Project edge points onto line to get endpoints
        line_normal = np.array([-line_dir[1], line_dir[0]])
        projections = (best_edge_points - line_pt) @ line_dir
        t_min, t_max = projections.min(), projections.max()
        fit_p1 = line_pt + t_min * line_dir
        fit_p2 = line_pt + t_max * line_dir
        if (np.linalg.norm(fit_p1 - p1) + np.linalg.norm(fit_p2 - p2)
                > np.linalg.norm(fit_p2 - p1) + np.linalg.norm(fit_p1 - p2)):
            fit_p1, fit_p2 = fit_p2, fit_p1

        # Residual: mean perpendicular distance
        diffs = best_edge_points - line_pt
        perp_dists = np.abs(diffs @ line_normal)
        residual = float(np.mean(perp_dists))

        # Confidence
        coverage = len(best_edge_points) / n_scanlines
        max_residual = max(scan_width * 0.2, 2.0)
        residual_score = max(0.0, 1.0 - residual / max_residual)
        confidence = min(1.0, coverage) * residual_score

        # Gradient strength
        ix = np.clip(np.round(best_edge_points[:, 0]).astype(int), 0, self._w - 1)
        iy = np.clip(np.round(best_edge_points[:, 1]).astype(int), 0, self._h - 1)
        grad_strength = float(np.mean(self._gradient[iy, ix]))

        # Reject fits where edge points have low gradient (noise, not real edges)
        image_grad_mean = float(np.mean(self._gradient))
        if grad_strength < max(min_gradient * 3.0, image_grad_mean * 3.0):
            return None

        return LineFitResult(
            p1=fit_p1,
            p2=fit_p2,
            edge_points=best_edge_points,
            residual=residual,
            confidence=confidence,
            n_edge_points=len(best_edge_points),
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
        preferred_side_point: np.ndarray | None = None,
        prefer_extreme_side: bool = False,
    ) -> np.ndarray:
        """Sample edges along perpendicular scanlines.

        For each scanline, finds all gradient peaks above threshold and picks
        the one closest to the center (predicted line position). This is
        CAD-guided: among multiple edges (chamfer, shadow, etc.), the one
        nearest the predicted geometry is the true feature boundary.
        """
        n_samples = max(7, int(2 * scan_width) + 1)
        offsets = np.linspace(-scan_width, scan_width, n_samples)
        t_values = np.linspace(0, 1, n_scanlines)

        edge_points: list[list[float]] = []
        dr = (2 * scan_width) / (n_samples - 1) if n_samples > 1 else 1.0

        for t in t_values:
            base = p1 + t * (p2 - p1)

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

            # Find all local maxima above threshold
            center_idx = n_samples // 2
            preferred_sign = None
            if preferred_side_point is not None:
                side_offset = float((preferred_side_point - base) @ normal)
                if abs(side_offset) > 1e-6:
                    preferred_sign = 1 if side_offset > 0.0 else -1
            peak_idx = self._find_closest_peak(
                grad_profile, min_gradient, center_idx,
                preferred_sign=preferred_sign,
                prefer_extreme_side=prefer_extreme_side,
            )
            if peak_idx is None:
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
    def _find_closest_peak(
        profile: np.ndarray,
        min_gradient: float,
        center: int,
        preferred_sign: int | None = None,
        prefer_extreme_side: bool = False,
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
        if n > 0 and profile[0] >= adaptive_threshold and profile[0] >= profile[min(1, n-1)]:
            peaks.append(0)
        if n > 1 and profile[-1] >= adaptive_threshold and profile[-1] >= profile[-2]:
            peaks.append(n - 1)

        if preferred_sign is not None and peaks:
            side_peaks = [
                j for j in peaks
                if (j - center) * preferred_sign > 0
            ]
            if side_peaks:
                if prefer_extreme_side:
                    return max(side_peaks, key=lambda j: abs(j - center))
                return min(side_peaks, key=lambda j: abs(j - center))

        if peaks:
            return min(peaks, key=lambda j: abs(j - center))

        # Fallback: any sample above adaptive threshold, closest to center
        above = [j for j in range(n) if profile[j] >= adaptive_threshold]
        if preferred_sign is not None and above:
            side_above = [
                j for j in above
                if (j - center) * preferred_sign > 0
            ]
            if side_above:
                if prefer_extreme_side:
                    return max(side_above, key=lambda j: abs(j - center))
                return min(side_above, key=lambda j: abs(j - center))
        if above:
            return min(above, key=lambda j: abs(j - center))
        return None


    @staticmethod
    def _select_offset_cluster(
        points: np.ndarray,
        p1: np.ndarray,
        direction: np.ndarray,
        preferred_side_point: np.ndarray | None = None,
        prefer_extreme_side: bool = False,
    ) -> np.ndarray:
        """Keep one coherent line-edge cluster in normal-offset space.

        Scanlines can hit several parallel edges in a printed window or at line
        caps. SVD over mixed clusters rotates the fit. Clustering by offset
        keeps the single physical edge before the final line fit.
        """
        if len(points) < 4:
            return points
        direction = np.array(direction, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            return points
        direction = direction / norm
        normal = np.array([-direction[1], direction[0]])
        t = (points - p1) @ direction
        base = p1 + t[:, None] * direction
        offsets = (points - base) @ normal

        order = np.argsort(offsets)
        sorted_offsets = offsets[order]
        clusters: list[np.ndarray] = []
        start = 0
        gap_px = 6.0
        for idx in range(1, len(sorted_offsets)):
            if sorted_offsets[idx] - sorted_offsets[idx - 1] > gap_px:
                clusters.append(order[start:idx])
                start = idx
        clusters.append(order[start:])

        support_fraction = 0.18 if preferred_side_point is not None else 0.08
        min_support = max(4, int(round(len(points) * support_fraction)))
        clusters = [cluster for cluster in clusters if len(cluster) >= min_support]
        if not clusters:
            return points

        preferred_sign = None
        if preferred_side_point is not None:
            center = np.mean(points, axis=0)
            side_offset = float((preferred_side_point - center) @ normal)
            if abs(side_offset) > 1e-6:
                preferred_sign = 1 if side_offset > 0.0 else -1

        if preferred_sign is not None:
            side_clusters = [
                cluster for cluster in clusters
                if float(np.median(offsets[cluster])) * preferred_sign > 0.0
            ]
            if side_clusters:
                clusters = side_clusters
                if prefer_extreme_side:
                    selected = max(
                        clusters,
                        key=lambda cluster: (
                            len(cluster),
                            abs(float(np.median(offsets[cluster]))),
                            -float(np.std(offsets[cluster])),
                        ),
                    )
                    return points[np.sort(selected)]

        selected = max(
            clusters,
            key=lambda cluster: (
                len(cluster),
                -abs(float(np.median(offsets[cluster]))),
                -float(np.std(offsets[cluster])),
            ),
        )
        return points[np.sort(selected)]

    @staticmethod
    def _fit_line_with_direction(
        points: np.ndarray, direction: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        direction = np.array(direction, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            direction = np.array([1.0, 0.0], dtype=np.float64)
        else:
            direction = direction / norm
        if direction[0] < 0:
            direction = -direction
        return direction, points.mean(axis=0)

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

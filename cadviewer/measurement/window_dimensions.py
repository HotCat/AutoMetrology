"""Stable hollow-window dimension measurements from fitted side lines.

The authoritative hollow-window width/height is the perpendicular separation
between opposite fitted edge lines.  Corner-to-corner segment lengths are kept
only as diagnostics because line intersections amplify small angular errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


PointCorrector = Callable[[np.ndarray], np.ndarray]


@dataclass
class OppositeLineThresholds:
    max_angle_diff_deg: float = 0.20
    max_std_px: float = 1.00
    max_range_px: float = 3.00
    min_samples: int = 25


@dataclass
class OppositeLineStats:
    separation_median_px: float
    separation_mean_px: float
    separation_std_px: float
    separation_min_px: float
    separation_max_px: float
    separation_range_px: float
    sample_count: int
    angle_difference_deg: float
    warnings: list[str]
    samples_px: list[float]

    def to_dict(self, pixel_size_mm: float) -> dict:
        return {
            "separation_median_px": float(self.separation_median_px),
            "separation_mean_px": float(self.separation_mean_px),
            "separation_std_px": float(self.separation_std_px),
            "separation_min_px": float(self.separation_min_px),
            "separation_max_px": float(self.separation_max_px),
            "separation_range_px": float(self.separation_range_px),
            "separation_median_mm": float(self.separation_median_px * pixel_size_mm),
            "separation_mean_mm": float(self.separation_mean_px * pixel_size_mm),
            "separation_std_mm": float(self.separation_std_px * pixel_size_mm),
            "separation_range_mm": float(self.separation_range_px * pixel_size_mm),
            "sample_count": int(self.sample_count),
            "angle_difference_deg": float(self.angle_difference_deg),
            "warnings": list(self.warnings),
            "samples_px": [float(v) for v in self.samples_px],
        }


@dataclass
class WindowLineDimensions:
    opposite_line_width_px: float
    opposite_line_height_px: float
    opposite_line_width_mm: float
    opposite_line_height_mm: float
    corner_segment_width_px: float
    corner_segment_height_px: float
    corner_segment_width_mm: float
    corner_segment_height_mm: float
    width_stats: OppositeLineStats
    height_stats: OppositeLineStats
    warnings: list[str]

    def distances_px(self) -> dict[str, float]:
        return {
            "opposite_line_width_px": float(self.opposite_line_width_px),
            "opposite_line_height_px": float(self.opposite_line_height_px),
            "corner_segment_width_px": float(self.corner_segment_width_px),
            "corner_segment_height_px": float(self.corner_segment_height_px),
            # Backward-compatible aliases now point to the authoritative values.
            "left_right_px": float(self.opposite_line_width_px),
            "top_bottom_px": float(self.opposite_line_height_px),
        }

    def distances_mm(self) -> dict[str, float]:
        return {
            "opposite_line_width_mm": float(self.opposite_line_width_mm),
            "opposite_line_height_mm": float(self.opposite_line_height_mm),
            "corner_segment_width_mm": float(self.corner_segment_width_mm),
            "corner_segment_height_mm": float(self.corner_segment_height_mm),
            # Backward-compatible aliases now point to the authoritative values.
            "width_mm": float(self.opposite_line_width_mm),
            "height_mm": float(self.opposite_line_height_mm),
        }

    def to_dict(self, pixel_size_mm: float) -> dict:
        return {
            "measurement_method": (
                "median perpendicular separation of opposite fitted backlight lines"
            ),
            "scale_source": "chessboard pixel_size_mm only",
            "opposite_line_width_px": float(self.opposite_line_width_px),
            "opposite_line_height_px": float(self.opposite_line_height_px),
            "opposite_line_width_mm": float(self.opposite_line_width_mm),
            "opposite_line_height_mm": float(self.opposite_line_height_mm),
            "corner_segment_width_px": float(self.corner_segment_width_px),
            "corner_segment_height_px": float(self.corner_segment_height_px),
            "corner_segment_width_mm": float(self.corner_segment_width_mm),
            "corner_segment_height_mm": float(self.corner_segment_height_mm),
            "corner_segment_is_diagnostic_only": True,
            "warnings": list(self.warnings),
            "width_pair": self.width_stats.to_dict(pixel_size_mm),
            "height_pair": self.height_stats.to_dict(pixel_size_mm),
        }


def normalize_line(line: tuple[float, float, float] | np.ndarray) -> tuple[float, float, float]:
    a, b, c = [float(v) for v in line]
    norm = float(np.hypot(a, b))
    if norm <= 1e-12:
        raise RuntimeError("Degenerate line equation")
    return a / norm, b / norm, c / norm


def line_intersection(
    line_a: tuple[float, float, float],
    line_b: tuple[float, float, float],
) -> np.ndarray:
    a1, b1, c1 = normalize_line(line_a)
    a2, b2, c2 = normalize_line(line_b)
    det = a1 * b2 - a2 * b1
    if abs(det) <= 1e-9:
        raise RuntimeError("Fitted window side lines are parallel")
    return np.array([
        (b1 * c2 - b2 * c1) / det,
        (c1 * a2 - c2 * a1) / det,
    ], dtype=np.float64)


def side_line_corners(
    lines: dict[str, tuple[float, float, float]],
    point_corrector: Optional[PointCorrector] = None,
) -> np.ndarray:
    corners = np.array([
        line_intersection(lines["left"], lines["top"]),
        line_intersection(lines["right"], lines["top"]),
        line_intersection(lines["right"], lines["bottom"]),
        line_intersection(lines["left"], lines["bottom"]),
    ], dtype=np.float64)
    if point_corrector is not None:
        corners = np.asarray(point_corrector(corners), dtype=np.float64)
    return corners


def corner_segment_distances(corners: np.ndarray) -> dict[str, float]:
    top_left, top_right, bottom_right, bottom_left = np.asarray(corners, dtype=np.float64)
    top_width = float(np.linalg.norm(top_right - top_left))
    bottom_width = float(np.linalg.norm(bottom_right - bottom_left))
    left_height = float(np.linalg.norm(bottom_left - top_left))
    right_height = float(np.linalg.norm(bottom_right - top_right))
    return {
        "corner_segment_width_px": (top_width + bottom_width) * 0.5,
        "corner_segment_height_px": (left_height + right_height) * 0.5,
        "top_width_px": top_width,
        "bottom_width_px": bottom_width,
        "left_height_px": left_height,
        "right_height_px": right_height,
    }


def measure_window_line_dimensions(
    lines: dict[str, tuple[float, float, float]],
    pixel_size_mm: float,
    *,
    central_fraction: float = 0.70,
    sample_count: int = 81,
    thresholds: Optional[OppositeLineThresholds] = None,
    point_corrector: Optional[PointCorrector] = None,
) -> WindowLineDimensions:
    """Measure width/height from opposite fitted line separation."""
    if pixel_size_mm <= 0:
        raise ValueError("pixel_size_mm must be positive")
    thresholds = thresholds or OppositeLineThresholds()
    central_fraction = float(np.clip(central_fraction, 0.20, 0.95))
    sample_count = max(int(sample_count), thresholds.min_samples)

    norm_lines = {name: normalize_line(line) for name, line in lines.items()}
    source_corners = side_line_corners(norm_lines)
    measurement_corners = (
        np.asarray(point_corrector(source_corners), dtype=np.float64)
        if point_corrector is not None else source_corners
    )
    corner = corner_segment_distances(measurement_corners)

    width_stats = _opposite_pair_stats(
        norm_lines["left"],
        norm_lines["right"],
        source_corners,
        span_corner_indices=(0, 3, 1, 2),
        central_fraction=central_fraction,
        sample_count=sample_count,
        thresholds=thresholds,
        point_corrector=point_corrector,
    )
    height_stats = _opposite_pair_stats(
        norm_lines["top"],
        norm_lines["bottom"],
        source_corners,
        span_corner_indices=(0, 1, 3, 2),
        central_fraction=central_fraction,
        sample_count=sample_count,
        thresholds=thresholds,
        point_corrector=point_corrector,
    )
    warnings = [
        f"width: {msg}" for msg in width_stats.warnings
    ] + [
        f"height: {msg}" for msg in height_stats.warnings
    ]
    return WindowLineDimensions(
        opposite_line_width_px=width_stats.separation_median_px,
        opposite_line_height_px=height_stats.separation_median_px,
        opposite_line_width_mm=width_stats.separation_median_px * pixel_size_mm,
        opposite_line_height_mm=height_stats.separation_median_px * pixel_size_mm,
        corner_segment_width_px=corner["corner_segment_width_px"],
        corner_segment_height_px=corner["corner_segment_height_px"],
        corner_segment_width_mm=corner["corner_segment_width_px"] * pixel_size_mm,
        corner_segment_height_mm=corner["corner_segment_height_px"] * pixel_size_mm,
        width_stats=width_stats,
        height_stats=height_stats,
        warnings=warnings,
    )


def _opposite_pair_stats(
    line_a: tuple[float, float, float],
    line_b: tuple[float, float, float],
    corrected_corners: np.ndarray,
    *,
    span_corner_indices: tuple[int, int, int, int],
    central_fraction: float,
    sample_count: int,
    thresholds: OppositeLineThresholds,
    point_corrector: Optional[PointCorrector],
) -> OppositeLineStats:
    a = np.array(line_a, dtype=np.float64)
    b = np.array(line_b, dtype=np.float64)
    if float(np.dot(a[:2], b[:2])) < 0.0:
        b = -b

    normal = a[:2] + b[:2]
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-12:
        normal = a[:2]
        normal_norm = float(np.linalg.norm(normal))
    normal = normal / normal_norm
    tangent = np.array([-normal[1], normal[0]], dtype=np.float64)

    dot = float(np.clip(abs(np.dot(a[:2], b[:2])), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(dot)))

    i0, i1, j0, j1 = span_corner_indices
    side_mid_1 = (corrected_corners[i0] + corrected_corners[i1]) * 0.5
    side_mid_2 = (corrected_corners[j0] + corrected_corners[j1]) * 0.5
    center = (side_mid_1 + side_mid_2) * 0.5
    projections = np.array([p @ tangent for p in corrected_corners], dtype=np.float64)
    span = float(np.max(projections) - np.min(projections))
    half = 0.5 * span * central_fraction
    offsets = np.linspace(-half, half, sample_count)

    samples = []
    for offset in offsets:
        point = center + tangent * offset
        sep = _line_separation_at_point(
            tuple(a), tuple(b), point, normal,
            point_corrector=point_corrector,
        )
        if sep is not None and np.isfinite(sep):
            samples.append(float(sep))

    warnings = []
    if len(samples) < thresholds.min_samples:
        warnings.append(
            f"too few valid samples ({len(samples)} < {thresholds.min_samples})"
        )
    if not samples:
        raise RuntimeError("No valid opposite-line separation samples")

    arr = np.asarray(samples, dtype=np.float64)
    std = float(np.std(arr))
    sep_range = float(np.max(arr) - np.min(arr))
    if angle_deg > thresholds.max_angle_diff_deg:
        warnings.append(
            f"opposite lines angle mismatch {angle_deg:.4f} deg "
            f"> {thresholds.max_angle_diff_deg:.4f} deg"
        )
    if std > thresholds.max_std_px:
        warnings.append(
            f"separation std {std:.4f}px > {thresholds.max_std_px:.4f}px"
        )
    if sep_range > thresholds.max_range_px:
        warnings.append(
            f"separation range {sep_range:.4f}px > {thresholds.max_range_px:.4f}px"
        )
    return OppositeLineStats(
        separation_median_px=float(np.median(arr)),
        separation_mean_px=float(np.mean(arr)),
        separation_std_px=std,
        separation_min_px=float(np.min(arr)),
        separation_max_px=float(np.max(arr)),
        separation_range_px=sep_range,
        sample_count=int(arr.size),
        angle_difference_deg=angle_deg,
        warnings=warnings,
        samples_px=[float(v) for v in arr],
    )


def _line_separation_at_point(
    line_a: tuple[float, float, float],
    line_b: tuple[float, float, float],
    point: np.ndarray,
    normal: np.ndarray,
    *,
    point_corrector: Optional[PointCorrector],
) -> Optional[float]:
    if point_corrector is not None:
        return _corrected_line_separation_at_point(line_a, line_b, point, normal, point_corrector)

    s_values = []
    for line in (line_a, line_b):
        a, b, c = line
        denom = a * normal[0] + b * normal[1]
        if abs(denom) <= 1e-9:
            return None
        s_values.append(-(a * point[0] + b * point[1] + c) / denom)
    return abs(float(s_values[1] - s_values[0]))


def _corrected_line_separation_at_point(
    line_a: tuple[float, float, float],
    line_b: tuple[float, float, float],
    corrected_point: np.ndarray,
    normal: np.ndarray,
    point_corrector: PointCorrector,
) -> Optional[float]:
    # The fitted lines live in source pixel coordinates.  For residual maps,
    # sample in source space near the corrected point, correct both line hits,
    # then measure their corrected-coordinate separation.
    source_point = np.asarray(corrected_point, dtype=np.float64)
    hits = []
    for line in (line_a, line_b):
        a, b, c = line
        denom = a * normal[0] + b * normal[1]
        if abs(denom) <= 1e-9:
            return None
        s = -(a * source_point[0] + b * source_point[1] + c) / denom
        hits.append(source_point + normal * s)
    corrected = np.asarray(point_corrector(np.vstack(hits)), dtype=np.float64)
    return float(np.linalg.norm(corrected[1] - corrected[0]))

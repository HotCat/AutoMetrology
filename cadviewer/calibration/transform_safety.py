"""Validation helpers for pixel-to-world measurement transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TransformSafety:
    safe: bool
    reason: str = ""
    min_scale: float = 0.0
    max_scale: float = 0.0
    anisotropy: float = 1.0


def _apply_projective(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    hom = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    mapped = (matrix @ hom.T).T
    denom = mapped[:, 2:3]
    safe = np.where(np.abs(denom) > 1e-12, denom, 1.0)
    return mapped[:, :2] / safe


def _sample_points(image_size: Optional[tuple[int, int]]) -> np.ndarray:
    if image_size is None:
        return np.array([[0.0, 0.0]], dtype=np.float64)
    width, height = image_size
    if width <= 0 or height <= 0:
        return np.array([[0.0, 0.0]], dtype=np.float64)
    xs = [0.0, width * 0.5, max(width - 1.0, 0.0)]
    ys = [0.0, height * 0.5, max(height - 1.0, 0.0)]
    return np.array([[x, y] for y in ys for x in xs], dtype=np.float64)


def validate_pixel_to_world_transform(
    matrix: np.ndarray,
    pixel_size_mm: float,
    image_size: Optional[tuple[int, int]] = None,
    max_scale_error: float = 0.03,
    max_anisotropy: float = 1.02,
    max_field_scale_change: float = 1.02,
) -> TransformSafety:
    """Reject transforms that would change metrology scale across the field.

    A telecentric measurement transform should be close to a similarity:
    rotation plus uniform scale. Homography or non-uniform affine terms are only
    accepted when their local pixel scale stays near the calibrated pixel size.
    """
    try:
        m = np.asarray(matrix, dtype=np.float64)
    except Exception:
        return TransformSafety(False, "not numeric")
    if m.shape != (3, 3) or not np.all(np.isfinite(m)):
        return TransformSafety(False, "invalid shape or non-finite values")

    expected = float(pixel_size_mm)
    if expected <= 0.0 or not np.isfinite(expected):
        return TransformSafety(False, "invalid pixel size")

    scales: list[float] = []
    anisotropies: list[float] = []
    for p in _sample_points(image_size):
        base = _apply_projective(m, p.reshape(1, 2))[0]
        dx = _apply_projective(m, (p + np.array([1.0, 0.0])).reshape(1, 2))[0] - base
        dy = _apply_projective(m, (p + np.array([0.0, 1.0])).reshape(1, 2))[0] - base
        jac = np.column_stack([dx, dy])
        try:
            sv = np.linalg.svd(jac, compute_uv=False)
        except np.linalg.LinAlgError:
            return TransformSafety(False, "singular local Jacobian")
        if len(sv) < 2 or np.min(sv) <= 0.0 or not np.all(np.isfinite(sv)):
            return TransformSafety(False, "invalid local scale")
        scales.extend([float(sv[0]), float(sv[1])])
        anisotropies.append(float(sv[0] / sv[1]))

    min_scale = min(scales)
    max_scale = max(scales)
    anisotropy = max(anisotropies)
    scale_low = expected * (1.0 - max_scale_error)
    scale_high = expected * (1.0 + max_scale_error)
    if min_scale < scale_low or max_scale > scale_high:
        return TransformSafety(
            False,
            f"scale outside pixel-size tolerance: {min_scale:.6g}..{max_scale:.6g}",
            min_scale,
            max_scale,
            anisotropy,
        )
    if anisotropy > max_anisotropy:
        return TransformSafety(
            False,
            f"non-uniform scale ratio {anisotropy:.4f}",
            min_scale,
            max_scale,
            anisotropy,
        )
    field_change = max_scale / min_scale
    if field_change > max_field_scale_change:
        return TransformSafety(
            False,
            f"field scale changes by {field_change:.4f}",
            min_scale,
            max_scale,
            anisotropy,
        )
    return TransformSafety(True, "", min_scale, max_scale, anisotropy)

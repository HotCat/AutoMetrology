"""
AffineAlignmentSolver — 2D affine transform estimation for telecentric imaging.

Provides methods for computing, composing, inverting, and applying 3x3 affine
matrices that map image pixel coordinates to CAD world coordinates.

The affine model (6 DOF):
  [x']   [a  b  tx] [x]
  [y'] = [c  d  ty] [y]
  [1 ]   [0  0   1] [1]

For telecentric imaging this is sufficient — no perspective distortion.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple


def identity() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def solve_from_centroids(
    src_centroid: np.ndarray, dst_centroid: np.ndarray
) -> np.ndarray:
    """Translation-only affine from src centroid to dst centroid."""
    T = identity()
    T[0, 2] = dst_centroid[0] - src_centroid[0]
    T[1, 2] = dst_centroid[1] - src_centroid[1]
    return T


def solve_from_bbox(
    src_bbox: Tuple[float, float, float, float],
    dst_bbox: Tuple[float, float, float, float],
) -> np.ndarray:
    """Scale + translation from bounding box alignment."""
    s_min_x, s_min_y, s_max_x, s_max_y = src_bbox
    d_min_x, d_min_y, d_max_x, d_max_y = dst_bbox

    s_w = s_max_x - s_min_x
    s_h = s_max_y - s_min_y
    d_w = d_max_x - d_min_x
    d_h = d_max_y - d_min_y

    if abs(s_w) < 1e-12 or abs(s_h) < 1e-12:
        return identity()

    # Uniform scale to avoid aspect ratio distortion
    scale = min(d_w / s_w, d_h / s_h)

    s_cx = (s_min_x + s_max_x) / 2
    s_cy = (s_min_y + s_max_y) / 2
    d_cx = (d_min_x + d_max_x) / 2
    d_cy = (d_min_y + d_max_y) / 2

    T = identity()
    T[0, 0] = scale
    T[1, 1] = scale
    T[0, 2] = d_cx - s_cx * scale
    T[1, 2] = d_cy - s_cy * scale
    return T


def solve_from_correspondences(
    src_points: np.ndarray, dst_points: np.ndarray
) -> np.ndarray:
    """
    Least-squares affine from N >= 3 point correspondences.

    src_points: (N, 2) source coordinates
    dst_points: (N, 2) destination coordinates
    Returns: 3x3 affine matrix
    """
    n = src_points.shape[0]
    if n < 3:
        # Fall back to translation for 1-2 points
        if n == 0:
            return identity()
        src_c = src_points.mean(axis=0)
        dst_c = dst_points.mean(axis=0)
        return solve_from_centroids(src_c, dst_c)

    # Build system: A @ [a, b, tx, c, d, ty]^T = b
    A = np.zeros((2 * n, 6), dtype=np.float64)
    b = np.zeros(2 * n, dtype=np.float64)

    for i in range(n):
        sx, sy = src_points[i]
        dx, dy = dst_points[i]
        A[2 * i] = [sx, sy, 1, 0, 0, 0]
        A[2 * i + 1] = [0, 0, 0, sx, sy, 1]
        b[2 * i] = dx
        b[2 * i + 1] = dy

    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    T = identity()
    T[0, 0] = result[0]
    T[0, 1] = result[1]
    T[0, 2] = result[2]
    T[1, 0] = result[3]
    T[1, 1] = result[4]
    T[1, 2] = result[5]
    return T


def solve_similarity(
    src_points: np.ndarray, dst_points: np.ndarray
) -> np.ndarray:
    """
    Least-squares similarity transform (rotation + uniform scale + translation).

    Constrains the affine to 4 DOF: scale s, rotation θ, translation (tx, ty).
    This is the correct physical model for telecentric imaging — no shear or
    non-uniform scaling.

    Model: [u -v tx; v u ty; 0 0 1] where u = s*cos(θ), v = s*sin(θ).

    src_points: (N, 2) source coordinates
    dst_points: (N, 2) destination coordinates
    Returns: 3x3 affine matrix (constrained)
    """
    n = src_points.shape[0]
    if n < 2:
        if n == 0:
            return identity()
        src_c = src_points.mean(axis=0)
        dst_c = dst_points.mean(axis=0)
        return solve_from_centroids(src_c, dst_c)

    # System: A @ [u, v, tx, ty]^T = b
    A = np.zeros((2 * n, 4), dtype=np.float64)
    b = np.zeros(2 * n, dtype=np.float64)

    for i in range(n):
        sx, sy = src_points[i]
        dx, dy = dst_points[i]
        A[2 * i] = [sx, -sy, 1, 0]
        A[2 * i + 1] = [sy, sx, 0, 1]
        b[2 * i] = dx
        b[2 * i + 1] = dy

    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    u, v, tx, ty = result

    T = identity()
    T[0, 0] = u
    T[0, 1] = -v
    T[0, 2] = tx
    T[1, 0] = v
    T[1, 1] = u
    T[1, 2] = ty
    return T


def solve_rigid_with_fixed_scale(
    src_points: np.ndarray, dst_points: np.ndarray, scale: float
) -> np.ndarray:
    """
    Least-squares rigid transform (rotation + translation) with fixed scale.

    For telecentric imaging, the scale is determined by pixel_size_mm and
    should not change during ICP refinement. This solver only updates
    rotation and translation.

    Model: [[scale*cos(θ), -scale*sin(θ), tx],
            [scale*sin(θ),  scale*cos(θ), ty],
            [0, 0, 1]]

    Solves for rotation θ and translation (tx, ty) with scale fixed.

    src_points: (N, 2) source coordinates
    dst_points: (N, 2) destination coordinates
    scale: fixed scale factor
    Returns: 3x3 affine matrix
    """
    n = src_points.shape[0]
    if n < 2:
        if n == 0:
            T = identity()
            T[0, 0] = scale
            T[1, 1] = scale
            return T
        src_c = src_points.mean(axis=0)
        dst_c = dst_points.mean(axis=0)
        T = identity()
        T[0, 0] = scale
        T[1, 1] = scale
        T[0, 2] = dst_c[0] - src_c[0] * scale
        T[1, 2] = dst_c[1] - src_c[1] * scale
        return T

    # Center points to isolate rotation
    src_centroid = src_points.mean(axis=0)
    dst_centroid = dst_points.mean(axis=0)
    src_centered = src_points - src_centroid
    dst_centered = dst_points - dst_centroid

    # Compute rotation using the correlation method
    # For fixed scale, we solve: scale*R(θ)*src_centered ≈ dst_centered
    # This is equivalent to: R(θ)*src_centered ≈ dst_centered/scale
    # The rotation can be found by:
    #   cos(θ) = (Σ(x_s*x_d + y_s*y_d)) / (Σ(x_s^2 + y_s^2))
    #   sin(θ) = (Σ(x_s*y_d - y_s*x_d)) / (Σ(x_s^2 + y_s^2))
    # where x_d, y_d are dst_centered/scale

    target_centered = dst_centered / scale

    x_s = src_centered[:, 0]
    y_s = src_centered[:, 1]
    x_d = target_centered[:, 0]
    y_d = target_centered[:, 1]

    numerator_cos = np.sum(x_s * x_d + y_s * y_d)
    numerator_sin = np.sum(x_s * y_d - y_s * x_d)
    denominator = np.sum(x_s ** 2 + y_s ** 2)

    if denominator < 1e-12:
        cos_theta = 1.0
        sin_theta = 0.0
    else:
        cos_theta = numerator_cos / denominator
        sin_theta = numerator_sin / denominator

    # Normalize to ensure cos² + sin² = 1
    norm = np.sqrt(cos_theta ** 2 + sin_theta ** 2)
    if norm > 1e-12:
        cos_theta /= norm
        sin_theta /= norm

    # Build rotation matrix with fixed scale
    T = identity()
    T[0, 0] = scale * cos_theta
    T[0, 1] = -scale * sin_theta
    T[1, 0] = scale * sin_theta
    T[1, 1] = scale * cos_theta

    # Compute translation: T @ src_centroid + t = dst_centroid
    # t = dst_centroid - T[:2, :2] @ src_centroid
    T[0, 2] = dst_centroid[0] - (T[0, 0] * src_centroid[0] + T[0, 1] * src_centroid[1])
    T[1, 2] = dst_centroid[1] - (T[1, 0] * src_centroid[0] + T[1, 1] * src_centroid[1])

    return T


def extract_scale(T: np.ndarray) -> float:
    """Extract uniform scale from a similarity or rigid transform."""
    u = T[0, 0]
    v = T[1, 0]
    return np.sqrt(u ** 2 + v ** 2)


def compose(transforms: List[np.ndarray]) -> np.ndarray:
    """Compose multiple affine transforms: T = T_n * ... * T_1."""
    result = identity()
    for T in transforms:
        result = T @ result
    return result


def invert(T: np.ndarray) -> np.ndarray:
    """Invert a 3x3 affine matrix."""
    return np.linalg.inv(T)


def apply(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Apply affine transform to Nx2 points.

    Returns transformed Nx2 points.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        return points
    n = points.shape[0]
    homogeneous = np.hstack([points, np.ones((n, 1))])
    transformed = (T @ homogeneous.T).T
    return transformed[:, :2]


def apply_projective(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Apply a 3x3 affine or projective transform to Nx2 points.

    Unlike apply(), this divides by the homogeneous coordinate and is therefore
    safe for homographies.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        return points
    n = points.shape[0]
    homogeneous = np.hstack([points, np.ones((n, 1), dtype=np.float64)])
    transformed = (np.asarray(T, dtype=np.float64) @ homogeneous.T).T
    denom = transformed[:, 2:3]
    safe = np.where(np.abs(denom) > 1e-12, denom, np.nan)
    return transformed[:, :2] / safe


def extract_params(T: np.ndarray) -> dict:
    """Extract human-readable affine parameters."""
    return {
        "scale_x": np.sqrt(T[0, 0] ** 2 + T[1, 0] ** 2),
        "scale_y": np.sqrt(T[0, 1] ** 2 + T[1, 1] ** 2),
        "rotation_deg": np.degrees(np.arctan2(T[1, 0], T[0, 0])),
        "tx": T[0, 2],
        "ty": T[1, 2],
        "shear": np.arctan2(T[0, 1], T[1, 1]) - np.arctan2(T[1, 0], T[0, 0]),
    }

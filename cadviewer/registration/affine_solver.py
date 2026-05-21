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

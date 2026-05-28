"""
ICPRegistrationEngine — Iterative Closest Point for 2D registration.

Uses point-to-point ICP with:
  - scipy.spatial.cKDTree for fast nearest-neighbor lookups
  - Rigid transform (rotation + translation) with scale FIXED from initial estimate
  - Outlier rejection by distance threshold

For telecentric imaging, scale is determined by pixel_size_mm and should not
change during ICP refinement. Only rotation and translation are refined.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from .affine_solver import solve_rigid_with_fixed_scale, extract_scale, apply, identity

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class ICPRegistrationEngine:
    """Point-to-point ICP with rigid transform estimation (scale fixed)."""

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        outlier_distance: float = 10.0,
    ) -> None:
        if not HAS_SCIPY:
            raise RuntimeError(
                "scipy is required for ICPRegistrationEngine. "
                "Install with: pip install scipy"
            )
        self._max_iterations = max_iterations
        self._tolerance = tolerance
        self._outlier_distance = outlier_distance

    def align(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        initial_transform: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Run ICP alignment with scale FIXED from initial transform.

        source_points: Nx2 CAD sample points (world coords)
        target_points: Mx2 image edge points (world coords)
        initial_transform: 3x3 affine (scale is extracted and kept fixed)

        Returns dict with:
          'transform': 3x3 affine matrix
          'iterations': int
          'final_error': float (mean squared residual)
          'converged': bool
          'correspondences': list of (src_idx, tgt_idx, distance)
        """
        if initial_transform is None:
            T = identity()
            fixed_scale = 1.0
        else:
            T = initial_transform.copy()
            fixed_scale = extract_scale(initial_transform)

        if len(source_points) < 3 or len(target_points) < 3:
            return {
                "transform": T,
                "iterations": 0,
                "final_error": float("inf"),
                "converged": False,
                "correspondences": [],
                "scale": fixed_scale,
            }

        target_tree = cKDTree(target_points)
        prev_error = float("inf")
        correspondences = []

        for iteration in range(self._max_iterations):
            # Transform source points using current estimate
            transformed = apply(T, source_points)

            # Find nearest neighbors
            distances, indices = target_tree.query(transformed)

            # Outlier rejection
            mask = distances < self._outlier_distance
            if mask.sum() < 3:
                break

            matched_src = source_points[mask]
            matched_tgt = target_points[indices[mask]]

            # Estimate rigid transform with scale FIXED from initial
            T_new = solve_rigid_with_fixed_scale(matched_src, matched_tgt, fixed_scale)

            # Compute error (MSE of transformed matched points)
            transformed_new = apply(T_new, source_points)
            error = float(np.mean(
                np.sum((transformed_new[mask] - matched_tgt) ** 2, axis=1)
            ))

            # Store correspondences
            correspondences = [
                (int(i), int(indices[i]), float(distances[i]))
                for i in range(len(mask))
                if mask[i]
            ]

            # Check convergence
            if abs(prev_error - error) < self._tolerance:
                T = T_new
                prev_error = error
                break

            T = T_new
            prev_error = error

        return {
            "transform": T,
            "iterations": iteration + 1,
            "final_error": prev_error,
            "converged": np.sqrt(prev_error) < 1.0,  # RMSE < 1mm
            "correspondences": correspondences,
            "scale": fixed_scale,
        }

"""
ICPRegistrationEngine — Iterative Closest Point for 2D affine registration.

Uses point-to-point ICP with:
  - scipy.spatial.cKDTree for fast nearest-neighbor lookups
  - numpy.linalg.lstsq for 6-DOF affine estimation per iteration
  - Outlier rejection by distance threshold

Designed for telecentric imaging where affine (not rigid) transform is needed.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from .affine_solver import solve_from_correspondences, apply, identity

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class ICPRegistrationEngine:
    """Point-to-point ICP with affine transform estimation."""

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
        Run ICP alignment.

        source_points: Nx2 CAD sample points (world coords)
        target_points: Mx2 image edge points (world coords after initial transform)
        initial_transform: 3x3 affine (default: identity)

        Returns dict with:
          'transform': 3x3 affine matrix
          'iterations': int
          'final_error': float (mean squared residual)
          'converged': bool
          'correspondences': list of (src_idx, tgt_idx, distance)
        """
        if initial_transform is None:
            T = identity()
        else:
            T = initial_transform.copy()

        if len(source_points) < 3 or len(target_points) < 3:
            return {
                "transform": T,
                "iterations": 0,
                "final_error": float("inf"),
                "converged": False,
                "correspondences": [],
            }

        target_tree = cKDTree(target_points)
        prev_error = float("inf")
        correspondences = []

        for iteration in range(self._max_iterations):
            # Transform source points
            transformed = apply(T, source_points)

            # Find nearest neighbors
            distances, indices = target_tree.query(transformed)

            # Outlier rejection
            mask = distances < self._outlier_distance
            if mask.sum() < 3:
                break

            matched_src = source_points[mask]
            matched_tgt = target_points[indices[mask]]

            # Estimate new affine
            T_new = solve_from_correspondences(matched_src, matched_tgt)

            # Compute error
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
            "converged": abs(prev_error) < self._outlier_distance,
            "correspondences": correspondences,
        }

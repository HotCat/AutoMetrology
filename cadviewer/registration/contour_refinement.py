"""
Lightweight contour refinement after coarse minAreaRect alignment.

Uses only outer contour points for ICP refinement — no internal features.
This prevents the local-minima problems caused by repetitive internal
geometry (circles, nested contours, parallel lines).

Iteration count is limited and outlier rejection is aggressive.
"""

from __future__ import annotations

import numpy as np

from . import affine_solver

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class ContourRefinementEngine:
    """Lightweight outer contour refinement using ICP on silhouette only."""

    def __init__(
        self,
        max_iterations: int = 30,
        tolerance: float = 1e-4,
        outlier_distance: float = 5.0,
    ) -> None:
        if not HAS_SCIPY:
            raise RuntimeError(
                "scipy is required for ContourRefinementEngine. "
                "Install with: pip install scipy"
            )
        self._max_iterations = max_iterations
        self._tolerance = tolerance
        self._outlier_distance = outlier_distance

    def refine(
        self,
        cad_contour: np.ndarray,
        img_world_points: np.ndarray,
        initial_transform: np.ndarray,
    ) -> dict:
        """Refine coarse alignment using outer contour ICP.

        Args:
            cad_contour: Mx2 float64 outer contour in CAD world coords
            img_world_points: Nx2 float64 image edge points in world coords
            initial_transform: 3x3 affine from coarse stage

        Returns:
            dict with 'transform', 'iterations', 'final_error', 'converged'
        """
        T = initial_transform.copy()
        fixed_scale = affine_solver.extract_scale(T)

        if len(cad_contour) < 3 or len(img_world_points) < 3:
            return {
                "transform": T,
                "iterations": 0,
                "final_error": float("inf"),
                "converged": False,
            }

        target_tree = cKDTree(img_world_points)
        prev_error = float("inf")

        for iteration in range(self._max_iterations):
            transformed = affine_solver.apply(T, cad_contour)
            dists, indices = target_tree.query(transformed)

            # Outlier rejection
            mask = dists < self._outlier_distance
            if mask.sum() < 3:
                break

            matched_src = cad_contour[mask]
            matched_tgt = img_world_points[indices[mask]]

            T_new = affine_solver.solve_rigid_with_fixed_scale(
                matched_src, matched_tgt, fixed_scale,
            )

            # Compute MSE of transformed matched points
            transformed_new = affine_solver.apply(T_new, cad_contour)
            error = float(np.mean(np.sum(
                (transformed_new[mask] - matched_tgt) ** 2, axis=1,
            )))

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
            "converged": np.sqrt(prev_error) < 1.0,
        }

"""
ConstrainedICP — ICP refinement bounded around a taught initial pose.

After the user teaches an initial pose, this engine refines it with strict
limits on translation (±5 mm), rotation (±2°), and scale change (±1%).
This prevents ICP from drifting to wrong local minima.
"""

from __future__ import annotations

import numpy as np

from . import affine_solver

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class ConstrainedICP:
    """ICP refinement with bounded drift from initial transform."""

    def __init__(
        self,
        max_iterations: int = 30,
        tolerance: float = 1e-4,
        outlier_distance: float = 5.0,
        max_translation: float = 5.0,
        max_rotation_deg: float = 2.0,
        max_scale_change: float = 0.01,
    ) -> None:
        if not HAS_SCIPY:
            raise RuntimeError("scipy required for ConstrainedICP")
        self._max_iterations = max_iterations
        self._tolerance = tolerance
        self._outlier_distance = outlier_distance
        self._max_translation = max_translation
        self._max_rotation_deg = max_rotation_deg
        self._max_scale_change = max_scale_change

    def refine(
        self,
        cad_contour: np.ndarray,
        img_world_points: np.ndarray,
        initial_transform: np.ndarray,
    ) -> dict:
        """Refine taught pose with constrained ICP.

        Args:
            cad_contour: Mx2 float64 CAD points in world coords.
            img_world_points: Nx2 float64 image edge points in world coords.
            initial_transform: 3x3 taught affine (CAD→image world).

        Returns:
            dict with 'transform', 'iterations', 'final_error', 'converged',
            and constraint diagnostics.
        """
        T = initial_transform.copy()

        if len(cad_contour) < 3 or len(img_world_points) < 3:
            return {
                "transform": T,
                "iterations": 0,
                "final_error": float("inf"),
                "converged": False,
                "clamped": False,
            }

        # Extract reference parameters from taught pose
        ref_params = affine_solver.extract_params(T)
        ref_scale = ref_params["scale_x"]
        ref_rot = ref_params["rotation_deg"]
        ref_tx = T[0, 2]
        ref_ty = T[1, 2]

        target_tree = cKDTree(img_world_points)
        prev_error = float("inf")
        clamped = False

        for iteration in range(self._max_iterations):
            transformed = affine_solver.apply(T, cad_contour)
            dists, indices = target_tree.query(transformed)

            mask = dists < self._outlier_distance
            if mask.sum() < 3:
                break

            matched_src = cad_contour[mask]
            matched_tgt = img_world_points[indices[mask]]

            # Solve with fixed scale from taught pose
            T_new = affine_solver.solve_rigid_with_fixed_scale(
                matched_src, matched_tgt, ref_scale,
            )

            # Check constraints
            new_tx = T_new[0, 2]
            new_ty = T_new[1, 2]
            dt = np.sqrt((new_tx - ref_tx) ** 2 + (new_ty - ref_ty) ** 2)
            if dt > self._max_translation:
                # Clamp translation
                factor = self._max_translation / dt
                T_new[0, 2] = ref_tx + (new_tx - ref_tx) * factor
                T_new[1, 2] = ref_ty + (new_ty - ref_ty) * factor
                clamped = True

            # Extract and check rotation
            cos_a = T_new[0, 0] / ref_scale
            sin_a = T_new[1, 0] / ref_scale
            new_rot = np.degrees(np.arctan2(sin_a, cos_a))
            rot_diff = abs(new_rot - ref_rot)
            if rot_diff > 180:
                rot_diff = 360 - rot_diff
            if rot_diff > self._max_rotation_deg:
                # Clamp rotation
                clamped_rot = ref_rot + np.sign(new_rot - ref_rot) * self._max_rotation_deg
                rad = np.radians(clamped_rot)
                T_new[0, 0] = ref_scale * np.cos(rad)
                T_new[0, 1] = -ref_scale * np.sin(rad)
                T_new[1, 0] = ref_scale * np.sin(rad)
                T_new[1, 1] = ref_scale * np.cos(rad)
                clamped = True

            # Compute error
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
            "clamped": clamped,
        }

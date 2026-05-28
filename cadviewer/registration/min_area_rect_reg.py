"""
MinAreaRect-based registration for stable global alignment.

Uses cv2.minAreaRect to compute coarse alignment between
CAD silhouette and image silhouette. This avoids ICP local minima
by using only global geometric properties (center, orientation, extent).

For telecentric imaging, the physical model is a similarity transform:
  uniform scale + rotation + translation (4 DOF).
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from . import affine_solver

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class MinAreaRectRegistration:
    """Register CAD to image using minAreaRect alignment."""

    def register(
        self,
        cad_points: np.ndarray,
        image_contour: np.ndarray,
        pixel_size_mm: float,
    ) -> tuple[np.ndarray, dict]:
        """Compute affine transform from CAD world to image world coordinates.

        Args:
            cad_points: Nx2 float64 in CAD world coords (mm)
            image_contour: Mx2 float64 in pixel coords
            pixel_size_mm: mm per pixel

        Returns:
            (3x3 affine matrix, debug_info dict)
            Returns (identity, {}) on failure.
        """
        if not HAS_CV2 or len(cad_points) < 3 or len(image_contour) < 3:
            return affine_solver.identity(), {}

        # Convert image contour to world coords
        img_world = image_contour.copy().astype(np.float64)
        img_world[:, 0] *= pixel_size_mm
        img_world[:, 1] *= -pixel_size_mm  # Y flip (CAD Y-up, image Y-down)

        # Compute minAreaRect for both point sets
        cad_rect = cv2.minAreaRect(cad_points.astype(np.float32))
        img_rect = cv2.minAreaRect(img_world.astype(np.float32))

        # Try all 4 angle combinations to resolve 90-degree ambiguity
        best_T: np.ndarray | None = None
        best_score = float("inf")
        best_info: dict = {}

        for cad_offset in (0, 90):
            for img_offset in (0, 90):
                T, info = self._try_alignment(
                    cad_points, img_world,
                    cad_rect, img_rect,
                    cad_offset, img_offset,
                )
                if T is None:
                    continue

                score = self._score_alignment(cad_points, img_world, T)
                if score < best_score:
                    best_score = score
                    best_T = T
                    best_info = info

        if best_T is None:
            return affine_solver.identity(), {}

        best_info["score"] = best_score
        best_info["cad_rect_raw"] = cad_rect
        best_info["img_rect_raw"] = img_rect
        return best_T, best_info

    # ── private helpers ───────────────────────────────────────────

    def _try_alignment(
        self,
        cad_points: np.ndarray,
        img_world: np.ndarray,
        cad_rect: tuple,
        img_rect: tuple,
        cad_offset: int,
        img_offset: int,
    ) -> tuple[np.ndarray | None, dict]:
        """Try one angle combination and return the resulting transform."""
        cad_center = np.array(cad_rect[0], dtype=np.float64)
        cad_size = np.array(cad_rect[1], dtype=np.float64)
        cad_angle = cad_rect[2]

        img_center = np.array(img_rect[0], dtype=np.float64)
        img_size = np.array(img_rect[1], dtype=np.float64)
        img_angle = img_rect[2]

        # Apply angle offsets and swap width/height for 90° rotations
        ca = cad_angle + cad_offset
        cs = cad_size.copy()
        if cad_offset == 90:
            cs = cs[::-1]

        ia = img_angle + img_offset
        iss = img_size.copy()
        if img_offset == 90:
            iss = iss[::-1]

        # Reject if aspect ratios are incompatible
        if cs[0] > 1e-6 and cs[1] > 1e-6 and iss[0] > 1e-6 and iss[1] > 1e-6:
            cad_ratio = max(cs[0], cs[1]) / max(min(cs[0], cs[1]), 1e-6)
            img_ratio = max(iss[0], iss[1]) / max(min(iss[0], iss[1]), 1e-6)
            if max(cad_ratio, img_ratio) > 1.5:
                ratio_err = abs(cad_ratio - img_ratio) / max(cad_ratio, img_ratio)
                if ratio_err > 0.5:
                    return None, {}

        # Scale from area ratio
        cad_area = abs(cs[0] * cs[1])
        img_area = abs(iss[0] * iss[1])
        if cad_area < 1e-6:
            return None, {}
        scale = np.sqrt(img_area / cad_area)

        # Rotation
        theta = np.radians(ia - ca)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Build similarity transform: scale * R(theta) + translation
        T = np.eye(3, dtype=np.float64)
        T[0, 0] = scale * cos_t
        T[0, 1] = -scale * sin_t
        T[1, 0] = scale * sin_t
        T[1, 1] = scale * cos_t

        # Translation: T @ cad_center == img_center
        T[0, 2] = img_center[0] - T[0, 0] * cad_center[0] - T[0, 1] * cad_center[1]
        T[1, 2] = img_center[1] - T[1, 0] * cad_center[0] - T[1, 1] * cad_center[1]

        info = {
            "cad_center": cad_center,
            "cad_size": cs,
            "cad_angle": cad_angle,
            "cad_angle_offset": cad_offset,
            "img_center": img_center,
            "img_size": iss,
            "img_angle": img_angle,
            "img_angle_offset": img_offset,
            "scale": scale,
            "rotation_deg": np.degrees(theta),
        }
        return T, info

    def _score_alignment(
        self, cad_points: np.ndarray, img_world: np.ndarray, T: np.ndarray,
    ) -> float:
        """Score alignment using mean squared chamfer distance."""
        if not HAS_SCIPY:
            # Fallback: just use transform consistency
            return float("inf")

        transformed = affine_solver.apply(T, cad_points)
        tree = cKDTree(img_world)
        dists, _ = tree.query(transformed)
        return float(np.mean(dists ** 2))

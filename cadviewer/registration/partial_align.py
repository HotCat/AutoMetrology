"""
PartialFOVAligner — coarse alignment for partial-FOV telecentric imaging.

Finds rotation + translation that best aligns CAD features with image edges,
assuming uniform scale (telecentric camera). The image may capture only a
subset of the full CAD geometry.

Strategy:
  1. Extract dominant orientations from CAD lines and image Hough lines.
  2. Enumerate rotations that align dominant orientations.
  3. For each candidate rotation, estimate translation in two stages:
     a. Centroid alignment (brings CAD within ~100mm of image).
     b. Iterative nearest-neighbor refinement (robust to partial overlap).
  4. Score by Gaussian-weighted distance (more discriminative than inlier
     count for grid-like structures).
"""

from __future__ import annotations

import numpy as np

from . import affine_solver

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _extract_cad_orientations(features_or_points, n_bins: int = 180) -> list[float]:
    """Extract dominant orientations from CAD features or point cloud.

    If features (list of CADFeature) are provided, uses line segment angles
    directly for accurate orientation detection. Otherwise falls back to
    local PCA on point neighborhoods.
    """
    # Try using feature geometry directly
    angles = []
    for f in (features_or_points if isinstance(features_or_points, list) else []):
        g = f.geometry if hasattr(f, 'geometry') else None
        if isinstance(g, dict) and 'x1' in g:
            dx = g['x2'] - g['x1']
            dy = g['y2'] - g['y1']
            if abs(dx) > 0.1 or abs(dy) > 0.1:
                angles.append(np.degrees(np.arctan2(dy, dx)))
        elif isinstance(g, dict) and 'cx' in g:
            # Circles contribute no orientation — skip
            pass

    if len(angles) >= 5:
        angles = np.array(angles)
        hist, bin_edges = np.histogram(angles, bins=n_bins, range=(-90, 90))
        peaks = np.argsort(hist)[-4:][::-1]
        dominant = [float(bin_edges[p]) for p in peaks if hist[p] > len(angles) * 0.03]
        return dominant if dominant else [0.0, 90.0]

    # Fallback: point-cloud based orientation using local PCA
    cad_points = features_or_points
    if not HAS_SCIPY or len(cad_points) < 10:
        return [0.0, 90.0]

    rng = np.random.default_rng(0)
    if len(cad_points) > 2000:
        idx = rng.choice(len(cad_points), 2000, replace=False)
        pts = cad_points[idx]
    else:
        pts = cad_points

    tree = cKDTree(pts)
    sample_idx = rng.choice(len(pts), min(200, len(pts)), replace=False)
    for i in sample_idx:
        neighbors_idx = tree.query_ball_point(pts[i], r=30.0)
        if len(neighbors_idx) < 3:
            continue
        neighbors = pts[neighbors_idx]
        centered = neighbors - neighbors.mean(axis=0)
        if len(centered) < 2:
            continue
        cov = centered.T @ centered / len(centered)
        eigvals, eigvecs = np.linalg.eigh(cov)
        dominant = eigvecs[:, -1]
        angle = np.degrees(np.arctan2(dominant[1], dominant[0]))
        angles.append(angle)

    if not angles:
        return [0.0, 90.0]

    angles_arr = np.array(angles)
    hist, bin_edges = np.histogram(angles_arr, bins=n_bins, range=(-90, 90))
    peaks = np.argsort(hist)[-4:][::-1]
    dominant = [float(bin_edges[p]) for p in peaks if hist[p] > len(angles_arr) * 0.02]
    return dominant if dominant else [0.0, 90.0]


def _extract_image_orientations(image: np.ndarray) -> list[float]:
    """Extract dominant orientations from image using Hough line detection."""
    if not HAS_CV2:
        return [0.0, 90.0]

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    linesP = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10,
    )
    if linesP is None or len(linesP) < 5:
        return [0.0, 90.0]

    angles = []
    for l in linesP:
        x1, y1, x2, y2 = l[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)

    angles = np.array(angles)
    hist, bin_edges = np.histogram(angles, bins=180, range=(-90, 90))
    peaks = np.argsort(hist)[-4:][::-1]
    dominant = [float(bin_edges[p]) for p in peaks if hist[p] > len(angles) * 0.05]
    return dominant if dominant else [0.0, 90.0]


def _orientation_candidates(
    cad_orientations: list[float],
    image_orientations: list[float],
    tolerance_deg: float = 5.0,
) -> list[float]:
    """Compute candidate rotations that align CAD and image orientations.

    For each pair of (cad_angle, img_angle), the rotation needed is
    img_angle - cad_angle. Deduplicates within tolerance.
    """
    candidates = set()
    for cad_a in cad_orientations:
        for img_a in image_orientations:
            rot = img_a - cad_a
            # Normalize to 0..360
            rot = rot % 360
            # Check if close to existing candidate
            is_dup = False
            for existing in list(candidates):
                diff = abs(rot - existing)
                diff = min(diff, 360 - diff)
                if diff < tolerance_deg:
                    is_dup = True
                    break
            if not is_dup:
                candidates.add(rot)

    # Always include 0, 90, 180, 270 as fallbacks
    for base in [0.0, 90.0, 180.0, 270.0]:
        is_dup = False
        for existing in list(candidates):
            diff = abs(base - existing)
            diff = min(diff, 360 - diff)
            if diff < tolerance_deg:
                is_dup = True
                break
        if not is_dup:
            candidates.add(base)

    return sorted(candidates)


class PartialFOVAligner:
    """Coarse alignment for partial-FOV telecentric imaging.

    Uses dominant orientation matching to narrow the rotation search,
    then two-stage translation estimation per candidate rotation.
    """

    def register(
        self,
        cad_points: np.ndarray,
        image_edge_points: np.ndarray,
        pixel_size_mm: float,
        image: np.ndarray | None = None,
        cad_features: list | None = None,
        initial_transform: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Compute rigid transform from CAD to image world coords.

        Args:
            cad_points: Nx2 float64 in CAD world coords (mm)
            image_edge_points: Mx2 float64 in pixel coords (x, y)
            pixel_size_mm: mm per pixel
            image: Optional original image for orientation extraction
            cad_features: Optional CAD features for orientation extraction
            initial_transform: Optional 3x3 anchor-derived transform.
                When provided, narrows rotation/translation search around it.

        Returns:
            (3x3 affine matrix, debug_info dict)
        """
        if not HAS_SCIPY or len(cad_points) < 3 or len(image_edge_points) < 3:
            return affine_solver.identity(), {}

        # Convert image edge points to world mm
        img_world = image_edge_points.copy().astype(np.float64)
        img_world[:, 0] *= pixel_size_mm
        img_world[:, 1] *= -pixel_size_mm

        # Subsample for speed
        rng = np.random.default_rng(42)
        if len(img_world) > 3000:
            idx = rng.choice(len(img_world), 3000, replace=False)
            img_world = img_world[idx]

        cad_sub = cad_points
        if len(cad_sub) > 5000:
            idx = rng.choice(len(cad_sub), 5000, replace=False)
            cad_sub = cad_sub[idx]

        img_centroid = img_world.mean(axis=0)

        if initial_transform is not None:
            # ── Anchor-constrained search ─────────────────────────
            anchor_angle = np.degrees(
                np.arctan2(initial_transform[1, 0], initial_transform[0, 0])
            )
            anchor_angle = anchor_angle % 360

            # Narrow rotation search: anchor angle ±5° in 0.5° steps
            candidates = []
            for delta in np.arange(-5.0, 5.25, 0.5):
                candidates.append((anchor_angle + delta) % 360)

            best_T: np.ndarray | None = None
            best_score = -1.0
            best_angle = 0.0

            for angle_deg in candidates:
                T, score = self._evaluate_rotation(
                    cad_sub, img_world, img_centroid, float(angle_deg),
                    anchor_translation=initial_transform[:2, 2],
                )
                if T is not None and score > best_score:
                    best_score = score
                    best_T = T
                    best_angle = float(angle_deg)
        else:
            # ── Orientation-based rotation candidates (unconstrained) ─
            cad_orients = _extract_cad_orientations(
                cad_features if cad_features is not None else cad_sub,
            )
            img_orients = (
                _extract_image_orientations(image) if image is not None
                else [0.0, 90.0]
            )
            candidates = _orientation_candidates(cad_orients, img_orients)

            # Also add dense grid for robustness (coarse 10 steps)
            for angle in range(0, 360, 10):
                if not any(min(abs(angle - c), 360 - abs(angle - c)) < 8
                           for c in candidates):
                    candidates.append(float(angle))
            candidates.sort()

            # ── Evaluate each candidate rotation ──────────────────
            best_T: np.ndarray | None = None
            best_score = -1.0
            best_angle = 0.0

            for angle_deg in candidates:
                T, score = self._evaluate_rotation(
                    cad_sub, img_world, img_centroid, float(angle_deg),
                )
                if T is not None and score > best_score:
                    best_score = score
                    best_T = T
                    best_angle = float(angle_deg)

            if best_T is None:
                return affine_solver.identity(), {}

            # ── Fine refinement: +/-3 deg around best, 0.5 deg steps
            for delta in np.arange(-3.0, 3.25, 0.5):
                if abs(delta) < 0.01:
                    continue
                angle = best_angle + delta
                T, score = self._evaluate_rotation(
                    cad_sub, img_world, img_centroid, angle,
                )
                if T is not None and score > best_score:
                    best_score = score
                    best_T = T
                    best_angle = angle

        if best_T is None:
            return affine_solver.identity(), {}

        # ── Translation fine-tuning: grid search with tighter sigma ────
        # The NN refinement can converge to a suboptimal translation for
        # partial FOV / grid structures. Do a local grid search to fix it.
        # Narrower grid when anchor-constrained (already near optimum).
        grid_range = 3.0 if initial_transform is not None else 5.0
        best_T = self._refine_translation(
            cad_sub, img_world, best_T, coarse_sigma=2.0, grid_range=grid_range,
        )

        # Recompute final score
        R_final = best_T[:2, :2]
        t_final = best_T[:2, 2]
        translated_cad = cad_sub @ R_final.T + t_final
        img_tree = cKDTree(img_world)
        dists_final, _ = img_tree.query(translated_cad)
        best_score = float(np.mean(np.exp(-dists_final ** 2 / (2 * 3.0 ** 2))))

        info = {
            "score": best_score,
            "rotation_deg": best_angle,
            "n_cad": len(cad_sub),
            "n_img": len(img_world),
            "orientation_candidates": candidates[:8],
        }
        return best_T, info

    def _refine_translation(
        self,
        cad_points: np.ndarray,
        img_world: np.ndarray,
        T: np.ndarray,
        coarse_sigma: float = 2.0,
        grid_range: float = 5.0,
    ) -> np.ndarray:
        """Fine-tune translation with a local grid search + descent.

        The iterative NN refinement can get trapped in local minima for
        grid-like structures or partial-FOV scenarios. This method does
        a coarse grid search then a fine grid to find the best translation.
        """
        R = T[:2, :2]
        t = T[:2, 2].copy()
        rotated_cad = cad_points @ R.T
        img_tree = cKDTree(img_world)

        best_t = t.copy()
        best_score = -1.0

        # Coarse grid: ±grid_range, 1mm steps
        for dx in np.arange(-grid_range, grid_range + 0.5, 1.0):
            for dy in np.arange(-grid_range, grid_range + 0.5, 1.0):
                t_test = t + np.array([dx, dy])
                translated = rotated_cad + t_test
                dists, _ = img_tree.query(translated)
                score = float(np.mean(np.exp(-dists ** 2 / (2 * coarse_sigma ** 2))))
                if score > best_score:
                    best_score = score
                    best_t = t_test

        # Fine grid: ±1.5mm around best, 0.5mm steps
        fine_sigma = coarse_sigma * 0.7
        for dx in np.arange(-1.5, 1.75, 0.5):
            for dy in np.arange(-1.5, 1.75, 0.5):
                t_test = best_t + np.array([dx, dy])
                translated = rotated_cad + t_test
                dists, _ = img_tree.query(translated)
                score = float(np.mean(np.exp(-dists ** 2 / (2 * fine_sigma ** 2))))
                if score > best_score:
                    best_score = score
                    best_t = t_test

        T_out = T.copy()
        T_out[0, 2] = best_t[0]
        T_out[1, 2] = best_t[1]
        return T_out

    def _evaluate_rotation(
        self,
        cad_points: np.ndarray,
        img_world: np.ndarray,
        img_centroid: np.ndarray,
        angle_deg: float,
        anchor_translation: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, float]:
        """Try one rotation with iterative translation estimation."""
        theta = np.radians(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

        rotated_cad = cad_points @ R.T

        if anchor_translation is not None:
            # Seed from anchor translation instead of centroid
            t = anchor_translation.copy()
        else:
            cad_centroid = rotated_cad.mean(axis=0)
            t = img_centroid - cad_centroid

        # Iterative NN refinement (5 iterations)
        for _ in range(5):
            translated_cad = rotated_cad + t
            cad_tree = cKDTree(translated_cad)
            dists, indices = cad_tree.query(img_world)

            # Use matches within 10mm for refinement — more robust for
            # partial FOV than percentage-based filtering
            close_mask = dists < 10.0
            if close_mask.sum() < 10:
                break

            close_img = img_world[close_mask]
            close_cad = translated_cad[indices[close_mask]]
            refinement = close_img - close_cad
            t_update = np.median(refinement, axis=0)
            t = t + t_update

            # Early stop if update is tiny
            if np.linalg.norm(t_update) < 0.1:
                break

        # Score: backward (CAD→img) Gaussian weighting (sigma=3mm).
        # Backward scoring measures how well CAD points land near image
        # edges — it naturally handles partial overlap because CAD points
        # outside the FOV get high distances (near-zero contribution).
        translated_cad = rotated_cad + t
        img_tree = cKDTree(img_world)
        dists_bwd, _ = img_tree.query(translated_cad)
        sigma = 3.0
        score = float(np.mean(np.exp(-dists_bwd ** 2 / (2 * sigma ** 2))))

        T = np.eye(3, dtype=np.float64)
        T[:2, :2] = R
        T[0, 2] = t[0]
        T[1, 2] = t[1]

        return T, score

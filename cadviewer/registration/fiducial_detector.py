"""
FiducialDetector -- registration via anchor holes + window features.

Uses specific identifiable features (anchor holes + dark rectangular windows)
for robust alignment, replacing edge-based matching that struggles with
repetitive grid structures.

Windows in CAD are typically drawn as individual LINE+ARC entities (not closed
polylines), so we locate them using the anchor-derived rough transform combined
with the CAD point cloud.

Workflow:
  1. Detect anchor circles via HoughCircles (reuses AnchorDetector).
  2. Compute rough similarity transform from anchors.
  3. Detect dark rectangular windows in the camera image.
  4. Use rough transform + CAD point cloud to find CAD window centers.
  5. Compute final similarity transform from all correspondences.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import affine_solver
from .anchor_detector import AnchorDetector, AnchorHeuristic, AnchorResult, AnchorMatch
from .image_extractor import ImageFeatureExtractor
from .cad_silhouette import RegistrationContourGenerator
from ..models.repository import FeatureRepository
from ..models.feature import FeatureType

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

try:
    import diplib as dip
    HAS_DIPLIB = True
except ImportError:
    HAS_DIPLIB = False


@dataclass
class WindowMatch:
    cad_center: np.ndarray      # (2,) in mm
    image_center: np.ndarray    # (2,) in pixels
    image_size: np.ndarray      # (2,) width, height in pixels
    confidence: float = 0.0


@dataclass
class FiducialResult:
    anchor_matches: list = field(default_factory=list)
    window_matches: list = field(default_factory=list)
    transform: np.ndarray | None = None
    confidence: float = 0.0
    image_windows: list = field(default_factory=list)
    image_circles: list = field(default_factory=list)


class WindowDetector:
    """Detect dark rounded-corner rectangular regions in camera image."""

    def detect(
        self,
        image: np.ndarray,
        pixel_size_mm: float,
        min_area_mm2: float = 20.0,
        max_area_mm2: float = 10000.0,
        min_aspect: float = 1.2,
        max_aspect: float = 10.0,
        fill_threshold: float = 0.60,
    ) -> list[dict]:
        """Find dark rectangular windows in image.

        Returns list of dicts with keys:
            cx, cy (center in pixels), width, height (pixels),
            angle (deg), area (px2), area_mm2, fill, contour
        """
        if not HAS_CV2:
            return []

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        windows = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue

            area_mm2 = area * (pixel_size_mm ** 2)
            if area_mm2 < min_area_mm2 or area_mm2 > max_area_mm2:
                continue

            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect

            if w < 1 or h < 1:
                continue

            aspect = max(w, h) / min(w, h)
            if aspect < min_aspect or aspect > max_aspect:
                continue

            rect_area = w * h
            if rect_area < 1:
                continue
            fill = area / rect_area
            if fill < fill_threshold:
                continue

            windows.append({
                "cx": float(cx),
                "cy": float(cy),
                "width": float(max(w, h)),
                "height": float(min(w, h)),
                "angle": float(angle),
                "area": float(area),
                "area_mm2": float(area_mm2),
                "fill": float(fill),
                "contour": cnt,
            })

        windows.sort(key=lambda w: w["area"], reverse=True)

        # Cluster by area to remove outliers (overall part, thin borders).
        if len(windows) >= 4:
            areas = np.array([w["area_mm2"] for w in windows])
            median_area = np.median(areas)
            filtered = [w for w in windows
                        if 0.3 * median_area < w["area_mm2"] < 2.0 * median_area]
            if len(filtered) >= 3:
                windows = filtered

        return windows


class FiducialDetector:
    """Full fiducial-based registration using anchor holes + windows."""

    def __init__(self) -> None:
        self._anchor_detector = AnchorDetector()
        self._window_detector = WindowDetector()
        self._silhouette_gen = RegistrationContourGenerator()

    @staticmethod
    def detect_circles(
        image: np.ndarray,
        min_radius_px: int = 3,
        max_radius_px: int = 200,
        min_dist_px: float = 10.0,
        pixel_size_mm: float = 0.1,
    ) -> list[dict]:
        """Detect circles using diplib watershed + MeasurementTool.

        Uses the same approach as the qualityAssurance project:
        1. Diplib watershed segmentation to find distinct objects
        2. MeasurementTool.Measure with Roundness to identify circular objects
        3. Gravity for center, Radius measurement for size

        Falls back to OpenCV HoughCircles when diplib is unavailable.
        """
        if not HAS_CV2:
            return []

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if HAS_DIPLIB:
            results = FiducialDetector._detect_circles_watershed(
                image, gray, min_radius_px, max_radius_px, pixel_size_mm,
            )
            if results:
                return results

        # Fallback: OpenCV HoughCircles
        return ImageFeatureExtractor.detect_circles_hough(
            image, dp=1.0, min_dist=int(min_dist_px),
            param1=100, param2=20,
            min_radius=min_radius_px, max_radius=max_radius_px,
        )

    @staticmethod
    def _detect_circles_watershed(
        image: np.ndarray,
        gray: np.ndarray,
        min_radius_px: int,
        max_radius_px: int,
        pixel_size_mm: float,
    ) -> list[dict]:
        """Diplib watershed + MeasurementTool circle detection."""
        h, w = gray.shape

        # Build diplib image (RGB if available, else grey).
        # Do NOT set pixel size — work in pixel units so that
        # SmallObjectsRemove and Radius use pixels consistently.
        if len(image.shape) == 3:
            rgb = image[:, :, ::-1].copy()
            dip_img = dip.Image(rgb)
            dip_img.SetColorSpace('sRGB')
        else:
            dip_img = dip.Image(np.array(gray, dtype=np.float32))

        img_g = dip.Gauss(dip_img, 0.4)
        img_grey = dip.ColorSpaceManager.Convert(img_g, 'grey')
        gm = dip.Norm(dip.GradientMagnitude(img_grey))
        gm = dip.Opening(dip.Closing(gm, 3), 3)
        wlab = dip.Watershed(
            gm, connectivity=1, maxDepth=3,
            flags={'correct', 'labels'},
        )
        wlab = dip.SmallObjectsRemove(wlab, 20)
        wlab = dip.EdgeObjectsRemove(wlab)

        msr = dip.MeasurementTool.Measure(
            wlab, wlab,
            ['Size', 'Roundness', 'Radius', 'Gravity'],
        )

        results = []
        for o in msr.Objects():
            vals = msr[o]
            roundness = vals['Roundness'][0]
            radius = vals['Radius'][0]
            gravity = vals['Gravity']
            cx, cy = float(gravity[0]), float(gravity[1])

            if roundness < 0.30:
                continue
            if radius < min_radius_px or radius > max_radius_px:
                continue

            results.append({
                "cx": cx, "cy": cy,
                "radius": float(radius),
                "roundness": float(roundness),
            })

        # Sort by roundness descending (most circular first)
        results.sort(key=lambda c: c["roundness"], reverse=True)
        return results

    def register(
        self,
        image: np.ndarray,
        cad_anchors: list[dict],
        cad_features: list,
        pixel_size_mm: float,
    ) -> FiducialResult:
        """Run fiducial-based registration."""
        result = FiducialResult()

        # ── Step 0: Generate CAD point cloud ────────────────────────
        try:
            cad_points = self._silhouette_gen.generate_point_cloud(
                cad_features, density=0.5,
            )
        except (KeyError, ValueError, TypeError):
            cad_points = self._fallback_point_cloud(cad_features)

        # ── Step 1: Detect circles and windows ────────────────────────
        image_windows = self._window_detector.detect(image, pixel_size_mm)
        result.image_windows = image_windows

        rough_T = None
        anchor_result: AnchorResult = AnchorResult()

        # ── Step 2: Try matching ALL CAD circles (not just anchors) ─────
        # When designated anchors are too close together for a stable
        # transform, fall back to matching all circular features.
        if cad_anchors:
            anchor_result = self._match_anchors_ransac(
                image, cad_anchors, pixel_size_mm,
            )
            result.image_circles = anchor_result.image_circles

            if anchor_result.matches:
                # Quality gate: only accept if matched anchors are well-spaced
                src = np.array([m.cad_position for m in anchor_result.matches])
                cad_span = np.sqrt(np.sum((src.max(0) - src.min(0)) ** 2))
                if cad_span > 20.0 and anchor_result.transform is not None:
                    rough_T = anchor_result.transform
                    result.anchor_matches = anchor_result.matches

        if rough_T is None:
            # Anchor matching failed — try matching all CAD circles
            all_cad_circles = self._extract_cad_circles(cad_features)
            if len(all_cad_circles) >= 3:
                circle_T = self._match_all_circles(
                    image, all_cad_circles, pixel_size_mm,
                )
                if circle_T is not None:
                    rough_T = circle_T

        # ── Step 3: Fallback to window-only registration ───────────────
        if rough_T is None and len(image_windows) >= 6:
            window_T = self._register_from_windows_only(
                cad_points, image_windows, pixel_size_mm,
            )
            if window_T is not None:
                rough_T = window_T
                # Try anchor refinement with window-derived transform
                if cad_anchors and not anchor_result.matches:
                    anchor_result = self._match_anchors_with_known_transform(
                        image, cad_anchors, rough_T, pixel_size_mm,
                    )
                    if anchor_result.matches:
                        result.anchor_matches = anchor_result.matches
                        result.image_circles = anchor_result.image_circles

        if rough_T is None:
            rough_T = affine_solver.identity()

        # ── Step 3.5: Refine translation using circle alignment ──
        # Circle RANSAC finds correct rotation/scale but wrong translation
        # when circles form a regular grid (translational symmetry).
        # Refine translation by grid-searching to maximize circle overlap.
        # NOTE: 180° orientation disambiguation is handled at strategy level,
        # not here — we only fix translation, keeping the RANSAC's rotation.
        if not np.allclose(rough_T, affine_solver.identity()):
            refined_T, _ = self._refine_translation_grid_search(
                image, cad_features, rough_T, pixel_size_mm,
            )
            if refined_T is not None:
                rough_T = refined_T

        # ── Step 4: Find CAD window centers using point cloud ─────
        # Only use window matches if they improve the transform.
        window_matches = []
        if image_windows and len(cad_points) >= 10 and HAS_SCIPY:
            window_matches = self._match_windows_via_pointcloud(
                cad_points, image_windows, rough_T, pixel_size_mm,
            )
        result.window_matches = window_matches

        # ── Step 5: Compute final transform ──────────────────────────
        # When circle RANSAC succeeded but produced no anchor matches,
        # use rough_T directly — window matches via point cloud centroids
        # are often inaccurate.
        if not result.anchor_matches and not np.allclose(
            rough_T, affine_solver.identity(),
        ):
            result.transform = rough_T
            result.confidence = 0.5
            # Clear bogus window matches
            result.window_matches = []
            return result

        all_src = []
        all_dst = []

        for m in anchor_result.matches:
            all_src.append(m.cad_position)
            img_world = m.image_position.copy()
            img_world[0] *= pixel_size_mm
            img_world[1] *= -pixel_size_mm
            all_dst.append(img_world)

        for m in window_matches:
            all_src.append(m.cad_center)
            img_world = m.image_center.copy()
            img_world[0] *= pixel_size_mm
            img_world[1] *= -pixel_size_mm
            all_dst.append(img_world)

        if len(all_src) >= 2:
            src_pts = np.array(all_src)
            dst_pts = np.array(all_dst)
            T = affine_solver.solve_similarity(src_pts, dst_pts)

            transformed = affine_solver.apply(T, src_pts)
            residuals = np.sqrt(np.sum((transformed - dst_pts) ** 2, axis=1))
            mean_res = float(np.mean(residuals))

            # If refined transform is worse than rough_T, keep rough_T
            if window_matches:
                rough_res = self._transform_rmse(rough_T, src_pts, dst_pts)
                if mean_res > rough_res * 2.0:
                    result.transform = rough_T
                    result.confidence = max(0.0, 1.0 - rough_res / 5.0)
                    return result

            result.transform = T
            result.confidence = max(0.0, 1.0 - mean_res / 5.0)

        elif rough_T is not None and not np.allclose(rough_T, affine_solver.identity()):
            result.transform = rough_T
            result.confidence = 0.5

        return result

    @staticmethod
    def _transform_rmse(
        T: np.ndarray, src: np.ndarray, dst: np.ndarray,
    ) -> float:
        transformed = affine_solver.apply(T, src)
        return float(np.sqrt(np.mean(np.sum((transformed - dst) ** 2, axis=1))))

    @staticmethod
    def _extract_cad_circles(cad_features: list) -> list[dict]:
        """Extract all circle features from CAD for matching."""
        circles = []
        for f in cad_features:
            if f.feature_type == FeatureType.CIRCLE:
                g = f.geometry
                if isinstance(g, dict) and "radius" in g and "cx" in g:
                    circles.append({
                        "cx": g["cx"], "cy": g["cy"],
                        "radius": g["radius"],
                        "handle": f.dxf_handle,
                    })
        return circles

    def _match_all_circles(
        self,
        image: np.ndarray,
        cad_circles: list[dict],
        pixel_size_mm: float,
    ) -> np.ndarray | None:
        """Match detected image circles to ALL CAD circles using RANSAC.

        Detects circles in the image via diplib watershed, then uses RANSAC
        to find the similarity transform that best aligns them with CAD circles.
        Works even when designated anchors are too close together.
        """
        if not HAS_CV2 or len(cad_circles) < 3:
            return None

        max_cad_r = max(c["radius"] for c in cad_circles)
        min_cad_r = min(c["radius"] for c in cad_circles)

        # Detect circles with broad radius range to catch all CAD circles
        all_circles = self.detect_circles(
            image,
            min_radius_px=max(int(min_cad_r / pixel_size_mm * 0.5), 3),
            max_radius_px=int(max_cad_r / pixel_size_mm * 2.0) + 1,
            pixel_size_mm=pixel_size_mm,
        )
        if len(all_circles) < 3:
            return None

        best_T = None
        best_score = -1.0

        # Build correspondences: for each detected circle, find which CAD
        # circles it could match (by radius)
        for radius_tol in (0.20, 0.35, 0.50):
            pairs = []  # (img_idx, cad_idx)
            for i, c in enumerate(all_circles):
                r_mm = c["radius"] * pixel_size_mm
                for j, cad in enumerate(cad_circles):
                    if abs(r_mm - cad["radius"]) / cad["radius"] < radius_tol:
                        pairs.append((i, j))

            if len(pairs) < 6:
                continue

            # RANSAC: pick 2 random pairs, compute transform, count inliers
            rng = np.random.default_rng(42)
            n_trials = min(len(pairs) * 3, 500)

            for _ in range(n_trials):
                idx = rng.choice(len(pairs), 2, replace=False)
                i1, j1 = pairs[idx[0]]
                i2, j2 = pairs[idx[1]]

                if j1 == j2 or i1 == i2:
                    continue

                img_pts = np.array([
                    [all_circles[i1]["cx"] * pixel_size_mm,
                     -all_circles[i1]["cy"] * pixel_size_mm],
                    [all_circles[i2]["cx"] * pixel_size_mm,
                     -all_circles[i2]["cy"] * pixel_size_mm],
                ])
                cad_pts = np.array([
                    [cad_circles[j1]["cx"], cad_circles[j1]["cy"]],
                    [cad_circles[j2]["cx"], cad_circles[j2]["cy"]],
                ])

                cad_spacing = np.sqrt(np.sum((cad_pts[1] - cad_pts[0]) ** 2))
                img_spacing = np.sqrt(np.sum((img_pts[1] - img_pts[0]) ** 2))
                if cad_spacing < 1.0 or img_spacing < 0.5:
                    continue

                T = affine_solver.solve_similarity(cad_pts, img_pts)
                s = affine_solver.extract_scale(T)
                if s < 0.5 or s > 2.0:
                    continue

                # Count inliers
                inlier_count = 0
                inlier_src = []
                inlier_dst = []
                for pi, pj in pairs:
                    cad_pt = np.array([[cad_circles[pj]["cx"],
                                        cad_circles[pj]["cy"]]])
                    img_world = np.array([[
                        all_circles[pi]["cx"] * pixel_size_mm,
                        -all_circles[pi]["cy"] * pixel_size_mm,
                    ]])
                    transformed = affine_solver.apply(T, cad_pt)
                    err = np.sqrt(np.sum((transformed - img_world) ** 2))
                    if err < 5.0:
                        inlier_count += 1
                        inlier_src.append(cad_pt[0])
                        inlier_dst.append(img_world[0])

                if inlier_count > best_score and inlier_count >= 3:
                    best_score = inlier_count
                    if len(inlier_src) >= 2:
                        best_T = affine_solver.solve_similarity(
                            np.array(inlier_src), np.array(inlier_dst),
                        )

            if best_T is not None and best_score >= 3:
                break

        return best_T

    def _match_anchors_ransac(
        self,
        image: np.ndarray,
        cad_anchors: list[dict],
        pixel_size_mm: float,
    ) -> AnchorResult:
        """Rotation-invariant anchor matching using the most prominent circles.

        Finds the largest detected circles in the image and matches them to
        CAD anchor circles by radius and spacing. Uses scale/rotation
        plausibility to pick the best match.
        """
        if not HAS_CV2 or len(cad_anchors) < 2:
            return AnchorResult()

        # Detect circles — use conservative parameters that reliably find
        # real anchor holes without too many false positives
        cad_radii = [a["radius"] for a in cad_anchors]
        min_cad_r = min(cad_radii)
        max_cad_r = max(cad_radii)

        all_circles = self.detect_circles(
            image,
            min_radius_px=max(int(min_cad_r / pixel_size_mm * 0.5), 3),
            max_radius_px=int(max_cad_r / pixel_size_mm * 1.5) + 1,
            min_dist_px=10,
            pixel_size_mm=pixel_size_mm,
        )

        if len(all_circles) < 2:
            return AnchorResult(image_circles=all_circles)

        # Filter circles by tight radius match to CAD anchors.
        # Telecentric cameras preserve geometry well, so 15% tolerance is generous.
        RADIUS_TOL = 0.15
        filtered = []
        for c in all_circles:
            r_mm = c["radius"] * pixel_size_mm
            best_err = float("inf")
            for a in cad_anchors:
                err = abs(r_mm - a["radius"]) / a["radius"]
                best_err = min(best_err, err)
            if best_err < RADIUS_TOL:
                filtered.append((c, best_err))

        if len(filtered) < 2:
            return AnchorResult(image_circles=all_circles)

        # Sort filtered circles by radius match quality (best first)
        filtered.sort(key=lambda x: x[1])

        # Build CAD anchor pairs
        cad_pairs = []
        for i in range(len(cad_anchors)):
            for j in range(i + 1, len(cad_anchors)):
                a1, a2 = cad_anchors[i], cad_anchors[j]
                dx = a2["cx"] - a1["cx"]
                dy = a2["cy"] - a1["cy"]
                spacing = np.sqrt(dx**2 + dy**2)
                if spacing > 1.0:
                    cad_pairs.append((i, j, spacing))

        # Try filtered circles against all CAD pairs
        top_n = min(len(filtered), 10)
        candidates = []

        for i in range(top_n):
            for j in range(i + 1, top_n):
                c1, err1 = filtered[i]
                c2, err2 = filtered[j]
                dx_px = c2["cx"] - c1["cx"]
                dy_px = c2["cy"] - c1["cy"]
                spacing_px = np.sqrt(dx_px**2 + dy_px**2)
                spacing_mm = spacing_px * pixel_size_mm

                for ci, cj, expected_mm in cad_pairs:
                    if abs(spacing_mm - expected_mm) > expected_mm * 0.25:
                        continue

                    r1_mm = c1["radius"] * pixel_size_mm
                    r2_mm = c2["radius"] * pixel_size_mm
                    cr1 = cad_anchors[ci]["radius"]
                    cr2 = cad_anchors[cj]["radius"]

                    if abs(r1_mm - cr1) / cr1 > 0.5:
                        continue
                    if abs(r2_mm - cr2) / cr2 > 0.5:
                        continue

                    # Try both orderings
                    for swap in (False, True):
                        if swap:
                            m1_cad, m2_cad = cj, ci
                            m1_img, m2_img = c2, c1
                        else:
                            m1_cad, m2_cad = ci, cj
                            m1_img, m2_img = c1, c2

                        m1 = AnchorMatch(
                            dxf_handle=cad_anchors[m1_cad]["handle"],
                            cad_position=np.array([
                                cad_anchors[m1_cad]["cx"],
                                cad_anchors[m1_cad]["cy"],
                            ]),
                            cad_radius=cad_anchors[m1_cad]["radius"],
                            image_position=np.array([
                                m1_img["cx"], m1_img["cy"],
                            ]),
                            image_radius=m1_img["radius"],
                        )
                        m2 = AnchorMatch(
                            dxf_handle=cad_anchors[m2_cad]["handle"],
                            cad_position=np.array([
                                cad_anchors[m2_cad]["cx"],
                                cad_anchors[m2_cad]["cy"],
                            ]),
                            cad_radius=cad_anchors[m2_cad]["radius"],
                            image_position=np.array([
                                m2_img["cx"], m2_img["cy"],
                            ]),
                            image_radius=m2_img["radius"],
                        )

                        # Score by verifying transform against image edges
                        src_pts = np.array([m1.cad_position, m2.cad_position])
                        dst_pts = np.array([
                            [m1.image_position[0]*pixel_size_mm,
                             -m1.image_position[1]*pixel_size_mm],
                            [m2.image_position[0]*pixel_size_mm,
                             -m2.image_position[1]*pixel_size_mm],
                        ])
                        T = affine_solver.solve_similarity(src_pts, dst_pts)
                        s = np.sqrt(T[0, 0]**2 + T[1, 0]**2)

                        # Skip obviously wrong scale
                        if s < 0.7 or s > 1.4:
                            continue

                        spacing_err = abs(spacing_mm - expected_mm) / expected_mm

                        # Verify by transforming image window centers to CAD
                        # and checking if they land near CAD features
                        verify_score = self._verify_transform(
                            T, image, pixel_size_mm,
                        )

                        score = (1.0 - spacing_err) * verify_score

                        candidates.append(([m1, m2], score, T))

        if not candidates:
            return AnchorResult(image_circles=all_circles)

        candidates.sort(key=lambda x: x[1], reverse=True)
        best_matches, best_score, best_T = candidates[0]

        # Compute proper world transform
        src_pts = np.array([m.cad_position for m in best_matches])
        dst_pts = np.array([
            [m.image_position[0]*pixel_size_mm,
             -m.image_position[1]*pixel_size_mm]
            for m in best_matches
        ])
        T = affine_solver.solve_similarity(src_pts, dst_pts)

        return AnchorResult(
            matches=best_matches,
            transform=T,
            confidence=best_score,
            image_circles=all_circles,
        )

    def _verify_transform(
        self,
        T: np.ndarray,
        image: np.ndarray,
        pixel_size_mm: float,
    ) -> float:
        """Score transform by checking if dark image regions align with
        regions where the inverse-transformed image has high gradient.

        Uses a quick check: transform image corners to CAD space, verify
        the transform maps the image within a reasonable CAD region.
        """
        if not HAS_CV2:
            return 1.0

        h, w = image.shape[:2]
        corners_img = np.array([
            [0, 0], [w, 0], [w, h], [0, h],
        ], dtype=np.float64)

        # Convert corners to world mm
        corners_world = corners_img.copy()
        corners_world[:, 0] *= pixel_size_mm
        corners_world[:, 1] *= -pixel_size_mm

        # Transform to CAD space
        inv_T = np.linalg.inv(T)
        corners_cad = affine_solver.apply(inv_T, corners_world)

        # Compute area of transformed image in CAD space (shoelace)
        n = len(corners_cad)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += corners_cad[i, 0] * corners_cad[j, 1]
            area -= corners_cad[j, 0] * corners_cad[i, 1]
        area = abs(area) / 2.0

        # Expected area based on pixel dimensions
        expected_area = (w * pixel_size_mm) * (h * pixel_size_mm)

        # Area ratio should be close to scale^2
        if expected_area < 1.0:
            return 0.0
        area_ratio = area / expected_area

        # For a reasonable transform, area_ratio should be close to 1.0
        # (distortion from rotation/scale is bounded)
        if area_ratio < 0.5 or area_ratio > 2.0:
            return 0.0

        return 1.0 - abs(area_ratio - 1.0)

    def _register_from_windows_only(
        self,
        cad_points: np.ndarray,
        image_windows: list[dict],
        pixel_size_mm: float,
    ) -> np.ndarray | None:
        """Register using image windows matched to dense CAD point regions.

        Finds rectangular high-density regions in the CAD point cloud
        and matches them to detected image windows by spatial pattern.
        """
        if not HAS_SCIPY or len(cad_points) < 100 or len(image_windows) < 6:
            return None

        # Get image window centers in world mm
        img_centers = np.array([
            [w["cx"] * pixel_size_mm, -w["cy"] * pixel_size_mm]
            for w in image_windows
        ])

        # Image window pairwise distances (invariant to rotation/scale)
        img_dists = []
        for i in range(len(img_centers)):
            for j in range(i + 1, len(img_centers)):
                img_dists.append(np.linalg.norm(img_centers[i] - img_centers[j]))
        img_dists = np.sort(img_dists)

        # Search for matching pattern in CAD point cloud.
        # Sample candidate origins, compute local window pattern, compare.
        tree = cKDTree(cad_points)
        rng = np.random.default_rng(42)

        # Sample candidate CAD positions from dense regions
        sample_pts = cad_points
        if len(sample_pts) > 5000:
            idx = rng.choice(len(sample_pts), 5000, replace=False)
            sample_pts = sample_pts[idx]

        # Compute local point density at each sample point
        best_T = None
        best_score = -1.0

        # Try a grid of rotations (0-360 in 10° steps) and a grid of
        # candidate translations (matching centroids)
        img_centroid = img_centers.mean(axis=0)
        cad_centroid = cad_points.mean(axis=0)

        for angle_deg in range(0, 360, 10):
            theta = np.radians(angle_deg)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

            # For each scale near 1.0
            for s100 in range(85, 116, 5):
                s = s100 / 100.0
                Rs = s * R

                # Transform image window centers to CAD space
                rotated = (img_centers - img_centroid) @ Rs.T
                cad_candidates = rotated + cad_centroid

                # Check how many CAD points are near each candidate window
                near_count = 0
                for cc in cad_candidates:
                    indices = tree.query_ball_point(cc, r=5.0)
                    if len(indices) > 5:
                        near_count += 1

                score = near_count / len(cad_candidates)
                if score > best_score:
                    best_score = score
                    # Build transform: CAD → image world
                    T = np.eye(3, dtype=np.float64)
                    T[:2, :2] = Rs
                    T[0, 2] = img_centroid[0] - Rs[0, 0] * cad_centroid[0] - Rs[0, 1] * cad_centroid[1]
                    T[1, 2] = img_centroid[1] - Rs[1, 0] * cad_centroid[0] - Rs[1, 1] * cad_centroid[1]
                    best_T = T

        if best_score > 0.3:
            return best_T
        return None

    def _match_anchors_with_known_transform(
        self,
        image: np.ndarray,
        cad_anchors: list[dict],
        known_T: np.ndarray,
        pixel_size_mm: float,
    ) -> AnchorResult:
        """Find anchor circles in image using a known transform to guide matching."""
        if not HAS_CV2:
            return AnchorResult()

        # Detect circles
        all_circles = self.detect_circles(
            image,
            min_radius_px=3, max_radius_px=200,
            min_dist_px=10,
            pixel_size_mm=pixel_size_mm,
        )

        if len(all_circles) < 2:
            return AnchorResult(image_circles=all_circles)

        # Project CAD anchors to image pixel coords
        inv_T = np.linalg.inv(known_T)
        matches = []

        for a in cad_anchors:
            cad_pt = np.array([[a["cx"], a["cy"]]])
            img_world = affine_solver.apply(known_T, cad_pt)[0]
            img_px = np.array([img_world[0] / pixel_size_mm,
                               -img_world[1] / pixel_size_mm])

            # Find detected circle closest to projected position
            best_c = None
            best_dist = float("inf")
            for c in all_circles:
                dist = np.sqrt((c["cx"] - img_px[0])**2 + (c["cy"] - img_px[1])**2)
                r_mm = c["radius"] * pixel_size_mm
                r_err = abs(r_mm - a["radius"]) / a["radius"]
                if dist < best_dist and r_err < 0.3:
                    best_dist = dist
                    best_c = c

            if best_c is not None and best_dist < 50:
                matches.append(AnchorMatch(
                    dxf_handle=a["handle"],
                    cad_position=np.array([a["cx"], a["cy"]]),
                    cad_radius=a["radius"],
                    image_position=np.array([best_c["cx"], best_c["cy"]]),
                    image_radius=best_c["radius"],
                    confidence=1.0 - best_dist / 50,
                ))

        return AnchorResult(
            matches=matches,
            transform=known_T,
            confidence=len(matches) / max(len(cad_anchors), 1),
            image_circles=all_circles,
        )

    def _refine_translation_grid_search(
        self,
        image: np.ndarray,
        cad_features: list,
        rough_T: np.ndarray,
        pixel_size_mm: float,
    ) -> tuple[np.ndarray | None, float]:
        """Refine translation of rough_T by grid-searching over circle alignment.

        Circle RANSAC finds correct rotation+scale but translation is ambiguous
        when circles form a regular grid. This method keeps the rotation+scale
        fixed and searches over translations to find the one where detected
        image circles best align with CAD circles.

        Returns:
            (T_refined, score) — the refined transform and its alignment score,
            or (None, 0.0) if refinement failed.
        """
        if not HAS_SCIPY or not HAS_CV2:
            return None, 0.0

        # Get CAD circles
        cad_circles = self._extract_cad_circles(cad_features)
        if len(cad_circles) < 3:
            return None, 0.0

        # Detect circles in image
        all_circles = self.detect_circles(
            image,
            min_radius_px=3,
            max_radius_px=200,
            pixel_size_mm=pixel_size_mm,
        )
        if len(all_circles) < 5:
            return None, 0.0

        # Image circle positions in world mm
        img_world = np.array([
            [c["cx"] * pixel_size_mm, -c["cy"] * pixel_size_mm]
            for c in all_circles
        ])
        img_tree = cKDTree(img_world)

        # CAD circle positions
        cad_pts = np.array([[c["cx"], c["cy"]] for c in cad_circles])

        # Extract rotation and scale from rough_T
        R = rough_T[:2, :2].copy()

        # Rotate CAD circles (without translation)
        rotated_cad = cad_pts @ R.T

        # Current translation
        t_current = rough_T[:2, 2].copy()

        # Estimate grid spacing from CAD circles
        if len(cad_pts) >= 4:
            from scipy.spatial.distance import pdist
            dists = pdist(cad_pts)
            # Use the most common short distance as grid spacing
            short_dists = dists[dists < 100]
            if len(short_dists) > 5:
                grid_spacing = float(np.median(short_dists))
            else:
                grid_spacing = 35.0
        else:
            grid_spacing = 35.0

        # Grid search: shift translation by ±2 grid spacings
        best_t = t_current.copy()
        best_score = -1.0

        search_range = grid_spacing * 2.0
        step = grid_spacing * 0.125  # 1/8 grid spacing

        img_w_mm = image.shape[1] * pixel_size_mm
        img_h_mm = image.shape[0] * pixel_size_mm

        for dx in np.arange(-search_range, search_range + step / 2, step):
            for dy in np.arange(-search_range, search_range + step / 2, step):
                t_test = t_current + np.array([dx, dy])
                translated = rotated_cad + t_test

                # Score: for each CAD circle, find nearest image circle
                dists_fwd, _ = img_tree.query(translated)
                # Only count CAD circles that land inside the image FOV
                in_fov = (translated[:, 0] > -10) & (translated[:, 0] < img_w_mm + 10) & \
                         (translated[:, 1] > -img_h_mm - 10) & (translated[:, 1] < 10)
                if in_fov.sum() < 3:
                    continue
                fwd_score = float(np.mean(np.exp(-dists_fwd[in_fov] ** 2 / (2 * 3.0 ** 2))))

                if fwd_score > best_score:
                    best_score = fwd_score
                    best_t = t_test

        # Fine search: ±2mm around best
        for dx in np.arange(-2.0, 2.25, 0.25):
            for dy in np.arange(-2.0, 2.25, 0.25):
                t_test = best_t + np.array([dx, dy])
                translated = rotated_cad + t_test
                dists_fwd, _ = img_tree.query(translated)
                in_fov = (translated[:, 0] > -10) & (translated[:, 0] < img_w_mm + 10) & \
                         (translated[:, 1] > -img_h_mm - 10) & (translated[:, 1] < 10)
                if in_fov.sum() < 3:
                    continue
                fwd_score = float(np.mean(np.exp(-dists_fwd[in_fov] ** 2 / (2 * 2.0 ** 2))))

                if fwd_score > best_score:
                    best_score = fwd_score
                    best_t = t_test

        # Rebuild transform
        T_out = np.eye(3, dtype=np.float64)
        T_out[:2, :2] = R
        T_out[0, 2] = best_t[0]
        T_out[1, 2] = best_t[1]

        return T_out, best_score

    def _refine_translation_with_windows(
        self,
        cad_features: list,
        image_windows: list[dict],
        rough_T: np.ndarray,
        pixel_size_mm: float,
    ) -> np.ndarray | None:
        """Refine translation of rough_T by matching image windows to CAD.

        Circle RANSAC finds correct rotation+scale but translation is ambiguous
        when circles form a regular grid. Windows provide unique spatial context
        to resolve the translational ambiguity.

        Strategy:
          1. Extract CAD window column centers (from rect features in CAD)
          2. Keep rotation+scale from rough_T
          3. Grid-search over translations to find the one that best aligns
             image window centers with CAD window centers
        """
        if not HAS_SCIPY or len(image_windows) < 3:
            return None

        # Extract CAD window column centers by finding rectangular regions
        cad_window_cols = self._extract_cad_window_centers(cad_features)
        if len(cad_window_cols) < 3:
            return None

        # Get image window centers in world mm
        img_win_centers = np.array([
            [w["cx"] * pixel_size_mm, -w["cy"] * pixel_size_mm]
            for w in image_windows
        ])

        # Filter to large windows only (the actual openings)
        img_areas = np.array([w["width"] * w["height"] for w in image_windows])
        median_area = np.median(img_areas)
        large_mask = img_areas > median_area * 0.5
        if large_mask.sum() < 2:
            large_mask = np.ones(len(image_windows), dtype=bool)
        img_win_centers = img_win_centers[large_mask]

        # Extract rotation and scale from rough_T
        params = affine_solver.extract_params(rough_T)
        scale = params["scale_x"]
        angle_rad = np.radians(params["rotation_deg"])
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[scale * cos_a, -scale * sin_a],
                       [scale * sin_a, scale * cos_a]])

        # Current translation from rough_T
        t_current = rough_T[:2, 2].copy()

        # Build a KD-tree of CAD window column centers
        cad_tree = cKDTree(cad_window_cols)

        # Grid search: shift translation by ±2 grid spacings
        # Estimate grid spacing from CAD window columns
        if len(cad_window_cols) >= 2:
            diffs = np.diff(np.sort(cad_window_cols[:, 0]))
            grid_x = float(np.median(diffs[diffs > 5]))
        else:
            grid_x = 70.0
        if len(cad_window_cols) >= 2:
            diffs_y = np.diff(np.sort(cad_window_cols[:, 1]))
            grid_y = float(np.median(diffs_y[diffs_y > 5])) if len(diffs_y[diffs_y > 5]) > 0 else 100.0
        else:
            grid_y = 100.0

        best_t = t_current.copy()
        best_score = -1.0

        # Coarse search: ±2 grid spacings, 0.5 grid steps
        for dx in np.arange(-2 * grid_x, 2 * grid_x + 1, grid_x * 0.25):
            for dy in np.arange(-2 * grid_y, 2 * grid_y + 1, grid_y * 0.25):
                t_test = t_current + np.array([dx, dy])

                # Transform CAD window centers to image world
                transformed = cad_window_cols @ R.T + t_test

                # Score: how many image window centers are close to a
                # transformed CAD window center
                tree_img = cKDTree(transformed)
                dists, _ = tree_img.query(img_win_centers)
                # Use Gaussian weighting (sigma = 5mm)
                score = float(np.mean(np.exp(-dists ** 2 / (2 * 5.0 ** 2))))

                if score > best_score:
                    best_score = score
                    best_t = t_test

        # Fine search: ±5mm around best, 0.5mm steps
        for dx in np.arange(-5.0, 5.25, 0.5):
            for dy in np.arange(-5.0, 5.25, 0.5):
                t_test = best_t + np.array([dx, dy])
                transformed = cad_window_cols @ R.T + t_test
                tree_img = cKDTree(transformed)
                dists, _ = tree_img.query(img_win_centers)
                score = float(np.mean(np.exp(-dists ** 2 / (2 * 3.0 ** 2))))

                if score > best_score:
                    best_score = score
                    best_t = t_test

        if best_score < 0.3:
            return None

        # Rebuild transform with refined translation
        T_out = np.eye(3, dtype=np.float64)
        T_out[:2, :2] = R
        T_out[0, 2] = best_t[0]
        T_out[1, 2] = best_t[1]
        return T_out

    @staticmethod
    def _extract_cad_window_centers(cad_features: list) -> np.ndarray:
        """Extract window column centers from CAD features.

        Finds rectangular openings by detecting vertical line pairs that
        form the window columns, then computes the center of each column.
        """
        from .cad_silhouette import RegistrationContourGenerator

        # Get all line features
        lines = []
        for f in cad_features:
            if f.feature_type == FeatureType.LINE:
                g = f.geometry
                if isinstance(g, dict) and 'x1' in g:
                    lines.append(g)

        if len(lines) < 8:
            return np.empty((0, 2))

        # Find unique X positions of vertical lines
        verticals = []
        for g in lines:
            dx = abs(g['x2'] - g['x1'])
            dy = abs(g['y2'] - g['y1'])
            if dy > dx * 3 and dy > 20:  # vertical line > 20mm
                x = (g['x1'] + g['x2']) / 2
                y_center = (g['y1'] + g['y2']) / 2
                verticals.append((x, y_center))

        if len(verticals) < 4:
            return np.empty((0, 2))

        # Cluster X positions to find unique columns
        xs = sorted(set(round(v[0], 0) for v in verticals))
        unique_x = [xs[0]]
        for x in xs[1:]:
            if abs(x - unique_x[-1]) > 5:
                unique_x.append(x)

        if len(unique_x) < 3:
            return np.empty((0, 2))

        # Find horizontal line Y positions to get top/bottom
        horizontals_y = []
        for g in lines:
            dx = abs(g['x2'] - g['x1'])
            dy = abs(g['y2'] - g['y1'])
            if dx > dy * 3 and dx > 20:  # horizontal line > 20mm
                y = (g['y1'] + g['y2']) / 2
                horizontals_y.append(y)

        if not horizontals_y:
            return np.empty((0, 2))

        y_min = min(horizontals_y)
        y_max = max(horizontals_y)
        y_center = (y_min + y_max) / 2

        # Window column centers: midpoints between consecutive vertical lines
        centers = []
        for i in range(len(unique_x) - 1):
            cx = (unique_x[i] + unique_x[i + 1]) / 2
            # Only include if the column width is reasonable (30-200mm)
            width = unique_x[i + 1] - unique_x[i]
            if 20 < width < 200:
                centers.append([cx, y_center])

        if not centers:
            return np.empty((0, 2))
        return np.array(centers, dtype=np.float64)

    def _match_windows_via_pointcloud(
        self,
        cad_points: np.ndarray,
        image_windows: list[dict],
        rough_T: np.ndarray,
        pixel_size_mm: float,
    ) -> list[WindowMatch]:
        """Match image windows to CAD using point cloud + rough transform.

        For each detected image window, project its center back to CAD space
        using the inverse of the rough transform. Then find the densest
        cluster of CAD points near that projected location — those points
        are the window outline in CAD.
        """
        if not HAS_SCIPY or rough_T is None:
            return []

        inv_T = np.linalg.inv(rough_T)
        cad_tree = cKDTree(cad_points)

        matches = []
        for iw in image_windows:
            # Convert image window center to world mm
            img_world = np.array([[
                iw["cx"] * pixel_size_mm,
                -iw["cy"] * pixel_size_mm,
            ]])

            # Project to CAD space using inverse transform
            cad_est = affine_solver.apply(inv_T, img_world)[0]

            # Search radius: 2x the larger window dimension in mm
            win_w_mm = iw["width"] * pixel_size_mm
            win_h_mm = iw["height"] * pixel_size_mm
            search_radius = max(win_w_mm, win_h_mm) * 1.5
            search_radius = max(search_radius, 10.0)

            # Find CAD points within search radius
            indices = cad_tree.query_ball_point(cad_est, search_radius)
            if len(indices) < 10:
                continue

            nearby = cad_points[indices]

            # The window outline points cluster along the edges of the window.
            # Find the centroid of nearby points that are on the outline
            # (not interior points from other features).
            # Strategy: use points that are near the boundary of the nearby
            # point cloud (within a band around the convex hull).
            center = nearby.mean(axis=0)

            # Compute distances from center — outline points are at
            # the periphery. Find the median distance to get the "radius"
            dists = np.sqrt(np.sum((nearby - center) ** 2, axis=1))
            median_dist = np.median(dists)

            # Keep points near the median distance (outline points)
            outline_mask = (dists > median_dist * 0.5) & (dists < median_dist * 1.5)
            if outline_mask.sum() < 5:
                # Fall back to using all nearby points
                cad_center = center
            else:
                cad_center = nearby[outline_mask].mean(axis=0)

            # Verify: the CAD center should be close to the estimated position
            offset = np.linalg.norm(cad_center - cad_est)
            if offset > search_radius:
                continue

            matches.append(WindowMatch(
                cad_center=cad_center,
                image_center=np.array([iw["cx"], iw["cy"]]),
                image_size=np.array([iw["width"], iw["height"]]),
                confidence=1.0 - offset / search_radius,
            ))

        return matches

    def _fallback_point_cloud(self, features: list) -> np.ndarray:
        """Simple point cloud extraction when silhouette generator fails."""
        points = []
        for f in features:
            g = f.geometry
            if not isinstance(g, dict):
                continue
            if f.feature_type == FeatureType.LINE:
                x1, y1, x2, y2 = g['x1'], g['y1'], g['x2'], g['y2']
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                n = max(int(length / 0.5), 2)
                for t in np.linspace(0, 1, n):
                    points.append([x1 + t*(x2-x1), y1 + t*(y2-y1)])
            elif f.feature_type == FeatureType.CIRCLE:
                if 'radius' in g and 'cx' in g:
                    cx, cy, r = g['cx'], g['cy'], g['radius']
                    n = max(int(2 * np.pi * r / 0.5), 8)
                    for t in np.linspace(0, 2*np.pi, n, endpoint=False):
                        points.append([cx + r*np.cos(t), cy + r*np.sin(t)])
            elif f.feature_type == FeatureType.ARC:
                if 'radius' in g and 'cx' in g:
                    cx, cy, r = g['cx'], g['cy'], g['radius']
                    a1 = np.radians(g.get('start_angle', 0))
                    a2 = np.radians(g.get('end_angle', 360))
                    n = max(int(abs(a2-a1) * r / 0.5), 4)
                    for t in np.linspace(a1, a2, n):
                        points.append([cx + r*np.cos(t), cy + r*np.sin(t)])
            elif f.feature_type == FeatureType.POLYLINE:
                pts = g.get('points', [])
                for i in range(len(pts) - 1):
                    x1, y1 = pts[i]
                    x2, y2 = pts[i+1]
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    n = max(int(length / 0.5), 2)
                    for t in np.linspace(0, 1, n):
                        points.append([x1 + t*(x2-x1), y1 + t*(y2-y1)])
        if not points:
            return np.empty((0, 2), dtype=np.float64)
        return np.array(points, dtype=np.float64)

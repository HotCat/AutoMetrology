"""
AnchorDetector -- detect and match anchor circles for grid-ambiguity resolution.

For CAD files with repetitive grid structures, the coarse aligner can get stuck
in local minima (matching the wrong grid period). Anchor circles at unique
positions (e.g., registration holes at the edges) break this ambiguity.

Workflow:
  1. HoughCircles detects circular features in the image.
  2. Pairs of detected circles are matched to known CAD anchor positions
     by spacing and radius constraints.
  3. Multiple candidate pairs are verified by checking transform consistency
     with additional detected circles.
  4. A rigid transform is computed from the best-matched correspondences.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import affine_solver
from .image_extractor import ImageFeatureExtractor
from ..models.repository import FeatureRepository
from ..models.feature import FeatureType

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@dataclass
class AnchorMatch:
    dxf_handle: str
    cad_position: np.ndarray     # (2,) in mm
    cad_radius: float            # mm
    image_position: np.ndarray   # (2,) in pixels
    image_radius: float          # pixels
    confidence: float = 0.0


@dataclass
class AnchorResult:
    matches: list[AnchorMatch] = field(default_factory=list)
    transform: np.ndarray | None = None
    confidence: float = 0.0
    image_circles: list[dict] = field(default_factory=list)


class AnchorDetector:
    """Detect anchor circles in an image and match to CAD anchor positions."""

    def detect_and_match(
        self,
        image: np.ndarray,
        cad_anchors: list[dict],
        pixel_size_mm: float,
    ) -> AnchorResult:
        if not HAS_CV2 or len(cad_anchors) < 2:
            return AnchorResult()

        cad_radii = [a["radius"] for a in cad_anchors]
        min_cad_r = min(cad_radii)
        max_cad_r = max(cad_radii)
        min_r_px = max(int(min_cad_r / pixel_size_mm * 0.6), 3)
        max_r_px = int(max_cad_r / pixel_size_mm * 1.8) + 1

        all_circles = ImageFeatureExtractor.detect_circles_hough(
            image,
            dp=1.0,
            min_dist=max(int(min_cad_r / pixel_size_mm * 3), 20),
            param1=100,
            param2=25,
            min_radius=min_r_px,
            max_radius=max_r_px,
        )

        if len(all_circles) < 2:
            return AnchorResult(image_circles=all_circles)

        filtered = []
        for c in all_circles:
            r_mm = c["radius"] * pixel_size_mm
            for a in cad_anchors:
                if abs(r_mm - a["radius"]) < a["radius"] * 0.5:
                    filtered.append(c)
                    break

        if len(filtered) < 2:
            return AnchorResult(image_circles=all_circles)

        candidates = self._find_all_pair_candidates(
            filtered, cad_anchors, pixel_size_mm,
        )

        if not candidates:
            return AnchorResult(image_circles=all_circles)

        # Resolve ambiguity by verifying each candidate against other
        # detected circles and CAD anchors.
        if len(candidates) > 1:
            candidates = self._verify_candidates(
                candidates, all_circles, cad_anchors, pixel_size_mm, image,
            )

        best_matches, best_score = candidates[0]

        src_pts = np.array([m.cad_position for m in best_matches])
        dst_px = np.array([m.image_position for m in best_matches])
        dst_pts = dst_px.copy()
        dst_pts[:, 0] *= pixel_size_mm
        dst_pts[:, 1] *= -pixel_size_mm

        T = affine_solver.solve_rigid_with_fixed_scale(src_pts, dst_pts, 1.0)

        return AnchorResult(
            matches=best_matches,
            transform=T,
            confidence=best_score,
            image_circles=all_circles,
        )

    def _find_all_pair_candidates(
        self,
        detected: list[dict],
        cad_anchors: list[dict],
        pixel_size_mm: float,
    ) -> list[tuple[list[AnchorMatch], float]]:
        """Find ALL candidate pairs of detected circles matching CAD spacing."""
        cad_pairs = []
        for i in range(len(cad_anchors)):
            for j in range(i + 1, len(cad_anchors)):
                a1, a2 = cad_anchors[i], cad_anchors[j]
                dx = a2["cx"] - a1["cx"]
                dy = a2["cy"] - a1["cy"]
                spacing_mm = np.sqrt(dx ** 2 + dy ** 2)
                if spacing_mm > 1.0:
                    cad_pairs.append((i, j, spacing_mm, dx, dy))

        if not cad_pairs:
            return []

        candidates = []
        for i in range(len(detected)):
            for j in range(i + 1, len(detected)):
                c1, c2 = detected[i], detected[j]
                dx_px = c2["cx"] - c1["cx"]
                dy_px = c2["cy"] - c1["cy"]
                spacing_px = np.sqrt(dx_px ** 2 + dy_px ** 2)
                spacing_mm = spacing_px * pixel_size_mm

                for ci, cj, expected_mm, cad_dx, cad_dy in cad_pairs:
                    if abs(spacing_mm - expected_mm) > expected_mm * 0.15:
                        continue
                    x_diff_px = abs(dx_px)
                    x_tolerance = max(spacing_px * 0.15, 10.0)
                    if x_diff_px > x_tolerance:
                        continue

                    r1_mm = c1["radius"] * pixel_size_mm
                    r2_mm = c2["radius"] * pixel_size_mm
                    cr1 = cad_anchors[ci]["radius"]
                    cr2 = cad_anchors[cj]["radius"]
                    r_err1 = abs(r1_mm - cr1) / max(cr1, 0.1)
                    r_err2 = abs(r2_mm - cr2) / max(cr2, 0.1)

                    # Skip if radius is way off
                    if r_err1 > 0.3 or r_err2 > 0.3:
                        continue

                    # Determine ordering (CAD Y-up vs pixel Y-down)
                    if dy_px * (-pixel_size_mm) * np.sign(cad_dy) > 0:
                        m1 = AnchorMatch(
                            dxf_handle=cad_anchors[ci]["handle"],
                            cad_position=np.array([cad_anchors[ci]["cx"],
                                                   cad_anchors[ci]["cy"]]),
                            cad_radius=cr1,
                            image_position=np.array([c1["cx"], c1["cy"]]),
                            image_radius=c1["radius"],
                        )
                        m2 = AnchorMatch(
                            dxf_handle=cad_anchors[cj]["handle"],
                            cad_position=np.array([cad_anchors[cj]["cx"],
                                                   cad_anchors[cj]["cy"]]),
                            cad_radius=cr2,
                            image_position=np.array([c2["cx"], c2["cy"]]),
                            image_radius=c2["radius"],
                        )
                    else:
                        m1 = AnchorMatch(
                            dxf_handle=cad_anchors[cj]["handle"],
                            cad_position=np.array([cad_anchors[cj]["cx"],
                                                   cad_anchors[cj]["cy"]]),
                            cad_radius=cr2,
                            image_position=np.array([c1["cx"], c1["cy"]]),
                            image_radius=c1["radius"],
                        )
                        m2 = AnchorMatch(
                            dxf_handle=cad_anchors[ci]["handle"],
                            cad_position=np.array([cad_anchors[ci]["cx"],
                                                   cad_anchors[ci]["cy"]]),
                            cad_radius=cr1,
                            image_position=np.array([c2["cx"], c2["cy"]]),
                            image_radius=c2["radius"],
                        )

                    spacing_err = abs(spacing_mm - expected_mm) / expected_mm
                    # Radius consistency between the two detected circles
                    r_consistency = 1.0 - abs(c1["radius"] - c2["radius"]) / max(
                        c1["radius"], c2["radius"], 1.0
                    )
                    score = (1.0 - spacing_err) * r_consistency

                    candidates.append(([m1, m2], score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def _verify_candidates(
        self,
        candidates: list[tuple[list[AnchorMatch], float]],
        all_circles: list[dict],
        cad_anchors: list[dict],
        pixel_size_mm: float,
        image: np.ndarray,
    ) -> list[tuple[list[AnchorMatch], float]]:
        """Re-rank candidates by checking transform consistency.

        For each candidate pair, compute the transform, then count how many
        other detected circles land near a CAD anchor position (and vice versa).
        """
        img_h, img_w = image.shape[:2]
        scored = []

        for matches, base_score in candidates:
            src_pts = np.array([m.cad_position for m in matches])
            dst_px = np.array([m.image_position for m in matches])
            dst_pts = dst_px.copy()
            dst_pts[:, 0] *= pixel_size_mm
            dst_pts[:, 1] *= -pixel_size_mm
            T = affine_solver.solve_rigid_with_fixed_scale(src_pts, dst_pts, 1.0)
            T_inv = np.linalg.inv(T)

            extra_evidence = 0
            for c in all_circles:
                # Skip circles already in the match
                is_matched = any(
                    abs(c["cx"] - m.image_position[0]) < 3.0 and
                    abs(c["cy"] - m.image_position[1]) < 3.0
                    for m in matches
                )
                if is_matched:
                    continue

                img_world = np.array([[c["cx"] * pixel_size_mm,
                                       c["cy"] * (-pixel_size_mm)]])
                cad_pt = affine_solver.apply(T_inv, img_world)[0]

                for a in cad_anchors:
                    dist = np.sqrt((cad_pt[0] - a["cx"]) ** 2 +
                                   (cad_pt[1] - a["cy"]) ** 2)
                    if dist < 5.0:
                        extra_evidence += 1
                        break

            boosted = base_score + extra_evidence * 0.15
            scored.append((matches, boosted))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


class AnchorHeuristic:
    """Auto-detect candidate anchor circles from CAD feature repository."""

    def find_anchor_candidates(
        self,
        repo: FeatureRepository,
        n: int = 4,
    ) -> list[dict]:
        circles = repo.features_by_type(FeatureType.CIRCLE)
        if not circles:
            return []

        all_features = list(repo.all_features())
        x_positions = []
        for f in all_features:
            g = f.geometry
            if isinstance(g, dict):
                if "cx" in g:
                    x_positions.append(g["cx"])
                elif "x1" in g:
                    x_positions.extend([g["x1"], g["x2"]])
        if not x_positions:
            return []

        x_min = min(x_positions)
        x_max = max(x_positions)
        x_range = x_max - x_min
        if x_range < 1.0:
            return []

        edge_threshold = x_range * 0.05

        candidates = []
        for f in circles:
            g = f.geometry
            r = g.get("radius", 0)
            cx = g["cx"]
            cy = g["cy"]

            if r < 3.0 and (cx - x_min < edge_threshold or x_max - cx < edge_threshold):
                handle = f.dxf_handle or ""
                candidates.append({
                    "handle": handle,
                    "cx": cx, "cy": cy,
                    "radius": r,
                    "edge_dist": min(cx - x_min, x_max - cx),
                })

        candidates.sort(key=lambda c: c["edge_dist"])
        return candidates[:n]

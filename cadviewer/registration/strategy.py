"""
Registration strategy pattern — pluggable registration methods.

Each strategy implements run_coarse / run_fine / run_full against a
RegistrationContext and returns typed result dataclasses.

Strategies:
  FullSilhouetteStrategy  — existing dense silhouette + minAreaRect + contour ICP
  ConvexHullStrategy      — sparse hull-based alignment for partial-FOV telecentric imaging
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..models.repository import FeatureRepository
from ..models.registration import RegistrationManager, RegistrationGroup
from ..models.feature import CADFeature, FeatureType
from .cad_silhouette import CADSilhouetteExtractor, RegistrationContourGenerator
from .cad_hull import CADHullGenerator, extract_image_hull
from .image_silhouette import ProductSilhouetteExtractor
from .min_area_rect_reg import MinAreaRectRegistration
from .contour_refinement import ContourRefinementEngine
from .image_extractor import ImageFeatureExtractor
from .partial_align import PartialFOVAligner
from .anchor_detector import AnchorDetector, AnchorHeuristic
from .fiducial_detector import FiducialDetector
from . import affine_solver

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

logger = logging.getLogger(__name__)


def _print(msg: str) -> None:
    print(f"[REG] {msg}")


# ── Data structures ──────────────────────────────────────────────────


@dataclass
class RegistrationContext:
    """Everything a strategy needs from the pipeline."""
    repo: FeatureRepository
    reg_manager: RegistrationManager
    group_id: str
    image_path: str
    pixel_size_mm: float
    debug_data: dict = field(default_factory=dict)
    anchor_handles: list[str] = field(default_factory=list)


@dataclass
class CoarseResult:
    transform: np.ndarray
    error: float
    stage: str = "coarse"


@dataclass
class FineResult:
    transform: np.ndarray
    error: float
    iterations: int = 0
    converged: bool = False
    stage: str = "fine"


@dataclass
class FullResult:
    transform: np.ndarray
    coarse_transform: np.ndarray
    coarse_error: float
    fine_error: float
    iterations: int = 0
    converged: bool = False
    stage: str = "full"


# ── Helpers ──────────────────────────────────────────────────────────


def _resolve_features(
    ctx: RegistrationContext,
) -> tuple[Optional[RegistrationGroup], list[CADFeature]]:
    group = ctx.reg_manager.get_group(ctx.group_id)
    if not group or not group.feature_ids:
        # Fallback: use all features from repo so registration still works
        all_feats = list(ctx.repo._features.values())
        if all_feats:
            _print(f"  Warning: empty group, using all {len(all_feats)} features")
        return None, all_feats
    features = [ctx.repo.get(fid) for fid in group.feature_ids]
    features = [f for f in features if f is not None]

    # If the group has too few features, expand to all features within
    # the group's bounding box. A group of 5 circles alone isn't enough
    # for robust registration — we need nearby lines/arcs for context.
    if len(features) < 20:
        bbox = group.bbox(ctx.repo)
        if bbox:
            min_x, min_y, max_x, max_y = bbox
            pad = max(max_x - min_x, max_y - min_y) * 0.3
            expanded = []
            for f in ctx.repo._features.values():
                g = f.geometry
                if not isinstance(g, dict):
                    continue
                ft = f.feature_type
                if ft == FeatureType.LINE:
                    pts = [(g['x1'], g['y1']), (g['x2'], g['y2'])]
                elif ft in (FeatureType.CIRCLE, FeatureType.ARC):
                    cx, cy, r = g['cx'], g['cy'], g.get('radius', 0)
                    pts = [(cx, cy)]
                elif ft == FeatureType.POLYLINE:
                    pts = [(p[0], p[1]) for p in g.get('points', [])]
                else:
                    continue
                for px, py in pts:
                    if min_x - pad <= px <= max_x + pad and min_y - pad <= py <= max_y + pad:
                        expanded.append(f)
                        break
            if len(expanded) > len(features):
                _print(f"  Expanded group from {len(features)} to {len(expanded)} features")
                features = expanded

    return group, features


def _compute_rmse(src: np.ndarray, tgt: np.ndarray, T: np.ndarray) -> float:
    if len(src) == 0 or len(tgt) == 0:
        return float("inf")
    transformed = affine_solver.apply(T, src)
    from scipy.spatial import cKDTree
    tree = cKDTree(tgt)
    dists, _ = tree.query(transformed)
    return float(np.sqrt(np.mean(dists ** 2)))


def _compute_image_rmse(
    img_points: np.ndarray, cad_points: np.ndarray, T: np.ndarray,
) -> float:
    """RMSE from image points to nearest transformed CAD point.

    Measures alignment quality from the image's perspective — robust
    to partial FOV where the image captures only a subset of the CAD.
    """
    if len(img_points) == 0 or len(cad_points) == 0:
        return float("inf")
    transformed_cad = affine_solver.apply(T, cad_points)
    from scipy.spatial import cKDTree
    tree = cKDTree(transformed_cad)
    dists, _ = tree.query(img_points)
    return float(np.sqrt(np.mean(dists ** 2)))


# ── Base class ───────────────────────────────────────────────────────


class RegistrationStrategy:
    """Base class for registration strategies."""

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def description(self) -> str:
        raise NotImplementedError

    def run_coarse(self, ctx: RegistrationContext) -> CoarseResult:
        raise NotImplementedError

    def run_fine(
        self, ctx: RegistrationContext, coarse_transform: np.ndarray,
    ) -> FineResult:
        raise NotImplementedError

    def run_full(self, ctx: RegistrationContext) -> FullResult:
        coarse = self.run_coarse(ctx)
        if coarse.error == float("inf"):
            return FullResult(
                transform=coarse.transform,
                coarse_transform=coarse.transform,
                coarse_error=coarse.error,
                fine_error=float("inf"),
            )
        fine = self.run_fine(ctx, coarse.transform)

        _print("=" * 60)
        _print("FULL REGISTRATION SUMMARY")
        _print(f"  Coarse RMSE: {coarse.error:.4f} mm")
        _print(f"  Refined RMSE: {np.sqrt(fine.error):.4f} mm  "
               f"({fine.iterations} iters, converged={fine.converged})")
        _print("=" * 60)

        return FullResult(
            transform=fine.transform,
            coarse_transform=coarse.transform,
            coarse_error=coarse.error,
            fine_error=fine.error,
            iterations=fine.iterations,
            converged=fine.converged,
            stage="full",
        )


# ── Full Silhouette Strategy (existing logic) ────────────────────────


class FullSilhouetteStrategy(RegistrationStrategy):
    """Dense silhouette + minAreaRect coarse + contour ICP fine.

    Works well when the entire product is visible in the image.
    """

    def __init__(self) -> None:
        self._silhouette_gen = RegistrationContourGenerator()
        self._img_silhouette = ProductSilhouetteExtractor()
        self._min_area_rect = MinAreaRectRegistration()
        self._refinement = ContourRefinementEngine(
            max_iterations=30, tolerance=1e-4, outlier_distance=5.0,
        )

    @property
    def name(self) -> str:
        return "Full Silhouette"

    @property
    def description(self) -> str:
        return "Dense silhouette alignment — best when entire product is visible"

    def run_coarse(self, ctx: RegistrationContext) -> CoarseResult:
        _print("=" * 60)
        _print("COARSE REGISTRATION (MinAreaRect Silhouette)")
        _print(f"  pixel_size_mm = {ctx.pixel_size_mm}")

        group, features = _resolve_features(ctx)
        if not features:
            _print("  ERROR: no features available")
            return CoarseResult(transform=affine_solver.identity(), error=float("inf"))

        sil_types = {"LINE", "POLYLINE", "ARC"}
        sil_count = sum(1 for f in features if f.feature_type.name in sil_types)
        _print(f"  Group: {group.name if group else 'all features'} ({len(features)} features, {sil_count} silhouette)")

        cad_points = self._silhouette_gen.generate_point_cloud(features, density=0.5)
        cad_contour = self._silhouette_gen.generate(features, density=0.5)

        if len(cad_points) < 3:
            _print("  ERROR: too few CAD silhouette points")
            return CoarseResult(transform=affine_solver.identity(), error=float("inf"))

        cad_centroid = cad_points.mean(axis=0)
        _print(f"  CAD: {len(cad_points)} pts, contour: "
               f"{len(cad_contour) if cad_contour is not None else 0} pts")

        image = ImageFeatureExtractor.load_image(ctx.image_path)
        _print(f"  Image: {image.shape[1]}x{image.shape[0]} px")

        mask, img_contour = self._img_silhouette.extract(image)
        if len(img_contour) < 3:
            _print("  ERROR: too few image silhouette points")
            return CoarseResult(transform=affine_solver.identity(), error=float("inf"))

        _print(f"  Image silhouette: {len(img_contour)} pts")

        T_coarse, rect_info = self._min_area_rect.register(
            cad_points, img_contour, ctx.pixel_size_mm,
        )
        if np.allclose(T_coarse, np.eye(3)):
            _print("  ERROR: minAreaRect registration failed")
            return CoarseResult(transform=affine_solver.identity(), error=float("inf"))

        params = affine_solver.extract_params(T_coarse)
        _print(f"  scale={params['scale_x']:.6f}  rot={params['rotation_deg']:.4f}deg")

        img_world = img_contour.copy().astype(np.float64)
        img_world[:, 0] *= ctx.pixel_size_mm
        img_world[:, 1] *= -ctx.pixel_size_mm

        error = _compute_rmse(cad_points, img_world, T_coarse)
        _print(f"  Coarse RMSE: {error:.4f} mm")

        image_edges = ImageFeatureExtractor.extract_edges(image)

        ctx.debug_data["coarse"] = {
            "cad_points": cad_points,
            "cad_contour": cad_contour,
            "image_edges": image_edges,
            "img_contour": img_contour,
            "img_contour_world": img_world,
            "mask": mask,
            "transform": T_coarse,
            "rect_info": rect_info,
            "pixel_size_mm": ctx.pixel_size_mm,
            "cad_centroid": cad_centroid,
            "image_path": ctx.image_path,
        }

        return CoarseResult(transform=T_coarse, error=error)

    def run_fine(
        self, ctx: RegistrationContext, coarse_transform: np.ndarray,
    ) -> FineResult:
        _print("-" * 60)
        _print("REFINEMENT (Outer Contour ICP)")

        coarse_params = affine_solver.extract_params(coarse_transform)
        _print(f"  Input: scale={coarse_params['scale_x']:.6f}  "
               f"rot={coarse_params['rotation_deg']:.4f}deg")

        group, features = _resolve_features(ctx)
        if not features:
            return FineResult(transform=coarse_transform, error=float("inf"))

        cad_contour = self._silhouette_gen.generate(features, density=0.5)
        if cad_contour is None or len(cad_contour) < 3:
            _print("  ERROR: no CAD silhouette contour")
            return FineResult(transform=coarse_transform, error=float("inf"))

        coarse_data = ctx.debug_data.get("coarse", {})
        img_world = coarse_data.get("img_contour_world")
        if img_world is None or len(img_world) < 3:
            _print("  ERROR: no cached image silhouette from coarse")
            return FineResult(transform=coarse_transform, error=float("inf"))

        _print(f"  CAD contour: {len(cad_contour)} pts, Image: {len(img_world)} pts")

        result = self._refinement.refine(cad_contour, img_world, coarse_transform)
        T_refined = result["transform"]

        refined_params = affine_solver.extract_params(T_refined)
        _print(f"  Refined ({result['iterations']} iters): "
               f"scale={refined_params['scale_x']:.6f}  "
               f"rot={refined_params['rotation_deg']:.4f}deg  "
               f"RMSE={np.sqrt(result['final_error']):.4f} mm")

        ctx.debug_data["fine"] = {
            "transform": T_refined,
            "iterations": result["iterations"],
            "error": result["final_error"],
            "converged": result["converged"],
            "cad_contour": cad_contour,
            "img_world": img_world,
        }

        return FineResult(
            transform=T_refined,
            error=result["final_error"],
            iterations=result["iterations"],
            converged=result["converged"],
        )


# ── Convex Hull Strategy (partial FOV) ───────────────────────────────


class ConvexHullStrategy(RegistrationStrategy):
    """Sparse convex hull alignment for partial-FOV telecentric imaging.

    Uses only the outer convex hull (20-50 vertices) instead of dense
    sampled contours (500+ points). Robust to incomplete silhouette,
    missing contour regions, and internal-structure local minima.

    Intended for global pose estimation, not precise metrology.
    """

    def __init__(self) -> None:
        self._hull_gen = CADHullGenerator()
        self._silhouette_gen = RegistrationContourGenerator()
        self._img_silhouette = ProductSilhouetteExtractor()
        self._partial_aligner = PartialFOVAligner()
        self._min_area_rect = MinAreaRectRegistration()
        self._refinement = ContourRefinementEngine(
            max_iterations=20, tolerance=1e-3, outlier_distance=15.0,
        )
        self._anchor_detector = AnchorDetector()
        self._anchor_heuristic = AnchorHeuristic()

    @property
    def name(self) -> str:
        return "Convex Hull (partial FOV)"

    @property
    def description(self) -> str:
        return "Sparse hull alignment — for partial-visibility telecentric imaging"

    def run_coarse(self, ctx: RegistrationContext) -> CoarseResult:
        _print("=" * 60)
        _print("COARSE REGISTRATION (Convex Hull / Partial FOV)")
        _print(f"  pixel_size_mm = {ctx.pixel_size_mm}")

        group, features = _resolve_features(ctx)
        if not features:
            _print("  ERROR: no features available")
            return CoarseResult(transform=affine_solver.identity(), error=float("inf"))

        # ── Step 1: CAD convex hull (sparse vertices) ─────────────
        cad_points = self._silhouette_gen.generate_point_cloud(features, density=0.5)
        cad_hull = self._hull_gen.generate_hull(features, density=0.5)

        if len(cad_hull) < 3:
            _print("  ERROR: too few CAD hull vertices")
            return CoarseResult(transform=affine_solver.identity(), error=float("inf"))

        cad_centroid = cad_points.mean(axis=0)
        _print(f"  CAD hull: {len(cad_hull)} vertices from {len(cad_points)} pts")

        # ── Step 2: Image silhouette + edges ──────────────────────
        image = ImageFeatureExtractor.load_image(ctx.image_path)
        _print(f"  Image: {image.shape[1]}x{image.shape[0]} px")

        mask, img_contour = self._img_silhouette.extract(image)
        if len(img_contour) < 3:
            _print("  ERROR: too few image contour points")
            return CoarseResult(transform=affine_solver.identity(), error=float("inf"))

        img_hull = extract_image_hull(img_contour)
        _print(f"  Image hull: {len(img_hull)} vertices from {len(img_contour)} pts")

        image_edges = ImageFeatureExtractor.extract_edges(image)
        _print(f"  Image edges: {len(image_edges)} pts")

        # ── Step 2.5: Anchor-based pre-alignment ───────────────────
        anchor_transform = None
        anchor_result = None

        cad_anchors = self._get_cad_anchors(ctx)
        if cad_anchors:
            anchor_result = self._anchor_detector.detect_and_match(
                image, cad_anchors, ctx.pixel_size_mm,
            )
            if anchor_result.matches:
                anchor_transform = anchor_result.transform
                _print(f"  Anchor alignment: {len(anchor_result.matches)} matches, "
                       f"confidence={anchor_result.confidence:.3f}")
                for m in anchor_result.matches:
                    _print(f"    {m.dxf_handle}: CAD ({m.cad_position[0]:.1f}, "
                           f"{m.cad_position[1]:.1f}) -> px ({m.image_position[0]:.0f}, "
                           f"{m.image_position[1]:.0f})")
            else:
                _print("  Anchor: no matches found")

        # ── Step 3: Partial-FOV alignment via rotation search ────
        T_coarse = affine_solver.identity()
        rect_info: dict = {}

        if len(image_edges) >= 10:
            if anchor_transform is not None:
                _print("  Using anchor transform as initial estimate")
            _print("  Trying partial-FOV aligner (rotation search)...")
            T_coarse, rect_info = self._partial_aligner.register(
                cad_points, image_edges, ctx.pixel_size_mm,
                image=image, cad_features=features,
                initial_transform=anchor_transform,
            )

        # Fallback to minAreaRect for full-FOV images
        if np.allclose(T_coarse, np.eye(3)):
            _print("  Partial aligner failed, trying minAreaRect fallback...")
            T_coarse, rect_info = self._min_area_rect.register(
                cad_points, img_contour, ctx.pixel_size_mm,
            )

        if np.allclose(T_coarse, np.eye(3)):
            _print("  ERROR: all alignment methods failed")
            return CoarseResult(transform=affine_solver.identity(), error=float("inf"))

        params = affine_solver.extract_params(T_coarse)
        _print(f"  scale={params['scale_x']:.6f}  rot={params['rotation_deg']:.4f}deg")

        # Report estimated scale deviation from 1.0 (pixel_size_mm error indicator)
        scale_deviation = (params['scale_x'] - 1.0) * 100.0
        _print(f"  Scale deviation from unity: {scale_deviation:+.2f}%")

        # Convert image hull to world coords
        img_hull_world = img_hull.copy().astype(np.float64)
        img_hull_world[:, 0] *= ctx.pixel_size_mm
        img_hull_world[:, 1] *= -ctx.pixel_size_mm

        # Also convert full contour for overlay compatibility
        img_world = img_contour.copy().astype(np.float64)
        img_world[:, 0] *= ctx.pixel_size_mm
        img_world[:, 1] *= -ctx.pixel_size_mm

        # Convert edge points to world coords for fine ICP
        img_edges_world = image_edges.copy().astype(np.float64)
        img_edges_world[:, 0] *= ctx.pixel_size_mm
        img_edges_world[:, 1] *= -ctx.pixel_size_mm

        # Error from image's perspective: how well does the image
        # edges fit the CAD (tolerant to partial overlap).
        # Use edge points filtered to those near transformed CAD (within 20mm)
        # to avoid inflating RMSE with noise/internal edges far from geometry.
        error = _compute_image_rmse(img_edges_world, cad_points, T_coarse)

        # Also compute a "clean" RMSE using only edge points near CAD geometry
        transformed_cad_check = affine_solver.apply(T_coarse, cad_points)
        from scipy.spatial import cKDTree as _KDTree
        cad_tree = _KDTree(transformed_cad_check)
        check_dists, _ = cad_tree.query(img_edges_world)
        near_mask = check_dists < 20.0
        if near_mask.sum() > 10:
            clean_rmse = float(np.sqrt(np.mean(check_dists[near_mask] ** 2)))
            _print(f"  Image→CAD RMSE (all edges): {error:.4f} mm  "
                   f"(from {len(img_edges_world)} edge pts)")
            _print(f"  Image→CAD RMSE (near geometry): {clean_rmse:.4f} mm  "
                   f"({near_mask.sum()}/{len(img_edges_world)} edges within 20mm)")
        else:
            _print(f"  Image→CAD RMSE: {error:.4f} mm")
        _print(f"  Aligner inlier fraction: {rect_info.get('score', 'N/A')}")
        _print(f"  Estimated scale: {rect_info.get('scale', 'N/A')}")

        # ── Store debug data ──────────────────────────────────────
        ctx.debug_data["coarse"] = {
            # Standard keys (overlay compatible)
            "cad_points": cad_points,
            "cad_contour": cad_hull,
            "image_edges": image_edges,
            "img_contour": img_contour,
            "img_contour_world": img_world,
            "mask": mask,
            "transform": T_coarse,
            "rect_info": rect_info,
            "pixel_size_mm": ctx.pixel_size_mm,
            "cad_centroid": cad_centroid,
            "image_path": ctx.image_path,
            # Hull-specific keys
            "cad_hull": cad_hull,
            "img_hull": img_hull,
            "img_hull_world": img_hull_world,
            "img_edges_world": img_edges_world,
            "strategy": "convex_hull",
        }

        # Store anchor debug data for overlay visualization
        if anchor_result is not None:
            ctx.debug_data["coarse"]["anchor"] = {
                "matches": [
                    {
                        "handle": m.dxf_handle,
                        "cad_position": m.cad_position.tolist(),
                        "image_position": m.image_position.tolist(),
                    }
                    for m in anchor_result.matches
                ],
                "image_circles": anchor_result.image_circles,
                "anchor_transform": anchor_result.transform,
                "confidence": anchor_result.confidence,
            }

        return CoarseResult(transform=T_coarse, error=error)

    def _get_cad_anchors(self, ctx: RegistrationContext) -> list[dict]:
        """Look up CAD anchor features from context handles or heuristic."""
        handles = ctx.anchor_handles

        if not handles:
            # Auto-detect heuristic
            candidates = self._anchor_heuristic.find_anchor_candidates(ctx.repo)
            if not candidates:
                return []
            handles = [c["handle"] for c in candidates]

        anchors = []
        for h in handles:
            feat = ctx.repo.get_by_handle(h.strip())
            if feat and feat.feature_type == FeatureType.CIRCLE:
                g = feat.geometry
                anchors.append({
                    "handle": h.strip(),
                    "cx": g["cx"], "cy": g["cy"],
                    "radius": g["radius"],
                })
        return anchors

    def run_fine(
        self, ctx: RegistrationContext, coarse_transform: np.ndarray,
    ) -> FineResult:
        _print("-" * 60)
        _print("REFINEMENT (Partial FOV ICP)")

        group, features = _resolve_features(ctx)
        if not features:
            return FineResult(transform=coarse_transform, error=float("inf"))

        # Use full CAD point cloud (includes internal features) for
        # robust partial-FOV ICP — sparse hull vertices are insufficient.
        cad_pts = self._silhouette_gen.generate_point_cloud(features, density=0.5)
        if len(cad_pts) < 3:
            _print("  ERROR: no CAD points for refinement")
            return FineResult(transform=coarse_transform, error=float("inf"))

        coarse_data = ctx.debug_data.get("coarse", {})
        # Use dense image edge points (not sparse contour) for ICP
        img_edges_world = coarse_data.get("img_edges_world")
        if img_edges_world is None or len(img_edges_world) < 3:
            img_edges_world = coarse_data.get("img_contour_world")
        if img_edges_world is None or len(img_edges_world) < 3:
            _print("  ERROR: no cached image data from coarse")
            return FineResult(transform=coarse_transform, error=float("inf"))

        _print(f"  CAD points: {len(cad_pts)}, Image edges: {len(img_edges_world)} pts")

        result = self._refinement.refine(cad_pts, img_edges_world, coarse_transform)
        T_refined = result["transform"]

        # Compare coarse vs refined from image's perspective.
        # Use edge points (more reliable than silhouette contour).
        img_eval = img_edges_world if len(img_edges_world) >= 10 else coarse_data.get("img_contour_world")
        if img_eval is None or len(img_eval) < 3:
            _print("  ERROR: no image data for evaluation")
            return FineResult(transform=coarse_transform, error=float("inf"))

        coarse_rmse = _compute_image_rmse(img_eval, cad_pts, coarse_transform)
        refined_rmse = _compute_image_rmse(img_eval, cad_pts, T_refined)

        if refined_rmse < coarse_rmse:
            T_final = T_refined
            refined_params = affine_solver.extract_params(T_refined)
            _print(f"  Refined ({result['iterations']} iters): "
                   f"RMSE {coarse_rmse:.4f} → {refined_rmse:.4f} mm (improved)")
            _print(f"  Refined scale: {refined_params['scale_x']:.6f}  "
                   f"rot: {refined_params['rotation_deg']:.4f}deg")
        else:
            T_final = coarse_transform
            _print(f"  ICP did not improve ({coarse_rmse:.4f} → {refined_rmse:.4f} mm), "
                   f"keeping coarse")

        final_mse = min(coarse_rmse, refined_rmse) ** 2

        ctx.debug_data["fine"] = {
            "transform": T_final,
            "iterations": result["iterations"],
            "error": final_mse,
            "converged": refined_rmse < coarse_rmse,
            "cad_contour": cad_pts,
            "img_world": img_edges_world,
            "strategy": "convex_hull",
        }

        return FineResult(
            transform=T_final,
            error=final_mse,
            iterations=result["iterations"],
            converged=refined_rmse < coarse_rmse,
        )


# ── Fiducial Strategy (anchor holes + windows) ────────────────────────


class FiducialStrategy(RegistrationStrategy):
    """Registration using anchor holes + dark rectangular windows.

    Identifies specific fiducial features (2 anchor holes + 6 windows)
    in both CAD and camera image, computes a similarity transform from
    matched correspondences. More robust than edge-based matching for
    parts with repetitive grid structures.
    """

    def __init__(self) -> None:
        self._fiducial_detector = FiducialDetector()
        self._anchor_heuristic = AnchorHeuristic()
        self._silhouette_gen = RegistrationContourGenerator()
        self._refinement = ContourRefinementEngine(
            max_iterations=20, tolerance=1e-3, outlier_distance=15.0,
        )

    @property
    def name(self) -> str:
        return "Fiducial-Based"

    @property
    def description(self) -> str:
        return "Anchor holes + window fiducials — robust for grid structures"

    def run_coarse(self, ctx: RegistrationContext) -> CoarseResult:
        _print("=" * 60)
        _print("COARSE REGISTRATION (Fiducial-Based)")
        _print(f"  pixel_size_mm = {ctx.pixel_size_mm}")

        group, features = _resolve_features(ctx)
        if not features:
            _print("  ERROR: no features available")
            return CoarseResult(transform=affine_solver.identity(), error=float("inf"))

        _print(f"  Group: {group.name if group else 'all features'} ({len(features)} features)")

        # Load in color — diplib watershed produces better circle detection
        # on color images than grayscale
        if HAS_CV2:
            image = cv2.imread(ctx.image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Cannot load image: {ctx.image_path}")
        else:
            image = ImageFeatureExtractor.load_image(ctx.image_path)
        _print(f"  Image: {image.shape[1]}x{image.shape[0]} px")

        # Get CAD anchors from full repo (not limited to group)
        cad_anchors = self._get_cad_anchors(ctx)
        _print(f"  CAD anchors: {len(cad_anchors)}")

        # Use ALL repo features for fiducial matching — the group may not
        # contain circles needed for circle RANSAC. The group defines the
        # area of interest; the detector needs all circles for matching.
        all_repo_features = list(ctx.repo._features.values())

        # Run fiducial detection and matching
        fiducial_result = self._fiducial_detector.register(
            image, cad_anchors, all_repo_features, ctx.pixel_size_mm,
        )

        _print(f"  Anchor matches: {len(fiducial_result.anchor_matches)}")
        _print(f"  Image windows found: {len(fiducial_result.image_windows)}")
        _print(f"  Window matches: {len(fiducial_result.window_matches)}")

        for m in fiducial_result.anchor_matches:
            _print(f"    Anchor {m.dxf_handle}: CAD ({m.cad_position[0]:.1f}, "
                   f"{m.cad_position[1]:.1f}) -> px ({m.image_position[0]:.0f}, "
                   f"{m.image_position[1]:.0f})")

        for i, m in enumerate(fiducial_result.window_matches):
            _print(f"    Window {i}: CAD ({m.cad_center[0]:.1f}, "
                   f"{m.cad_center[1]:.1f}) -> px ({m.image_center[0]:.0f}, "
                   f"{m.image_center[1]:.0f})")

        T_coarse = fiducial_result.transform
        if T_coarse is None:
            _print("  ERROR: fiducial matching failed")
            return CoarseResult(transform=affine_solver.identity(), error=float("inf"))

        params = affine_solver.extract_params(T_coarse)
        _print(f"  scale={params['scale_x']:.6f}  rot={params['rotation_deg']:.4f}deg")
        scale_deviation = (params['scale_x'] - 1.0) * 100.0
        _print(f"  Scale deviation from unity: {scale_deviation:+.2f}%")
        _print(f"  Confidence: {fiducial_result.confidence:.3f}")

        # Compute RMSE against edge points for consistency with other strategies
        image_edges = ImageFeatureExtractor.extract_edges(image)
        cad_points = self._silhouette_gen.generate_point_cloud(features, density=0.5)

        img_edges_world = image_edges.copy().astype(np.float64)
        img_edges_world[:, 0] *= ctx.pixel_size_mm
        img_edges_world[:, 1] *= -ctx.pixel_size_mm

        # ── Orientation disambiguation + translation refinement ──
        # Circle RANSAC has 180° ambiguity on regular grids.
        # Try both orientations and pick the one with better edge overlap.
        T_coarse = self._disambiguate_orientation(
            cad_points, img_edges_world, T_coarse, ctx.pixel_size_mm,
            image=image, cad_features=features,
        )

        error = _compute_image_rmse(img_edges_world, cad_points, T_coarse)
        _print(f"  Edge RMSE: {error:.4f} mm")

        # Fiducial correspondence RMSE (transform quality metric)
        all_src = []
        all_dst = []
        for m in fiducial_result.anchor_matches:
            all_src.append(m.cad_position)
            iw = m.image_position.copy()
            iw[0] *= ctx.pixel_size_mm
            iw[1] *= -ctx.pixel_size_mm
            all_dst.append(iw)
        for m in fiducial_result.window_matches:
            all_src.append(m.cad_center)
            iw = m.image_center.copy()
            iw[0] *= ctx.pixel_size_mm
            iw[1] *= -ctx.pixel_size_mm
            all_dst.append(iw)

        if all_src:
            transformed = affine_solver.apply(T_coarse, np.array(all_src))
            residuals = np.sqrt(np.sum((transformed - np.array(all_dst)) ** 2, axis=1))
            fid_rmse = float(np.sqrt(np.mean(residuals ** 2)))
            _print(f"  Fiducial RMSE: {fid_rmse:.4f} mm ({len(all_src)} correspondences)")

        # Extract image silhouette for debug overlay compatibility
        from .image_silhouette import ProductSilhouetteExtractor
        img_silhouette = ProductSilhouetteExtractor()
        mask, img_contour = img_silhouette.extract(image)
        img_world = img_contour.copy().astype(np.float64)
        img_world[:, 0] *= ctx.pixel_size_mm
        img_world[:, 1] *= -ctx.pixel_size_mm

        ctx.debug_data["coarse"] = {
            "cad_points": cad_points,
            "image_edges": image_edges,
            "img_contour": img_contour,
            "img_contour_world": img_world,
            "img_edges_world": img_edges_world,
            "mask": mask,
            "transform": T_coarse,
            "pixel_size_mm": ctx.pixel_size_mm,
            "cad_centroid": cad_points.mean(axis=0) if len(cad_points) > 0 else np.zeros(2),
            "image_path": ctx.image_path,
            "strategy": "fiducial",
            # Fiducial-specific debug
            "fiducial": {
                "anchor_matches": [
                    {
                        "handle": m.dxf_handle,
                        "cad_position": m.cad_position.tolist(),
                        "image_position": m.image_position.tolist(),
                    }
                    for m in fiducial_result.anchor_matches
                ],
                "window_matches": [
                    {
                        "cad_center": m.cad_center.tolist(),
                        "image_center": m.image_center.tolist(),
                        "confidence": m.confidence,
                    }
                    for m in fiducial_result.window_matches
                ],
                "image_windows": [
                    {"cx": w["cx"], "cy": w["cy"],
                     "width": w["width"], "height": w["height"]}
                    for w in fiducial_result.image_windows
                ],
                "image_circles": fiducial_result.image_circles,
                "confidence": fiducial_result.confidence,
            },
        }

        return CoarseResult(transform=T_coarse, error=error)

    def _get_cad_anchors(self, ctx: RegistrationContext) -> list[dict]:
        handles = ctx.anchor_handles
        if not handles:
            candidates = self._anchor_heuristic.find_anchor_candidates(ctx.repo)
            if not candidates:
                return []
            handles = [c["handle"] for c in candidates]

        anchors = []
        for h in handles:
            feat = ctx.repo.get_by_handle(h.strip())
            if feat and feat.feature_type == FeatureType.CIRCLE:
                g = feat.geometry
                anchors.append({
                    "handle": h.strip(),
                    "cx": g["cx"], "cy": g["cy"],
                    "radius": g["radius"],
                })
        return anchors

    def _refine_translation_edges(
        self,
        cad_points: np.ndarray,
        img_edges_world: np.ndarray,
        T_coarse: np.ndarray,
        pixel_size_mm: float,
    ) -> tuple[np.ndarray, float]:
        """Fix translation by maximizing edge overlap (rotation+scale fixed).

        Circle RANSAC on regular grids produces correct rotation+scale but
        ambiguous translation (shifting by one grid spacing gives identical
        circle matches).  This method keeps the linear part of T_coarse and
        searches for the translation that best aligns CAD geometry with
        image edges.

        Uses a two-stage grid search: coarse ±100mm at 4mm steps, then
        fine ±8mm at 0.5mm steps.  Scoring is Gaussian-weighted nearest-
        neighbor distance (sigma=2mm) between transformed CAD points and
        image edge points, filtered to the image FOV.

        Returns:
            (T_refined, score) — the refined transform and its edge overlap score.
        """
        from scipy.spatial import cKDTree

        if len(cad_points) < 10 or len(img_edges_world) < 10:
            return T_coarse, 0.0

        # Decompose T_coarse: keep rotation+scale, discard translation
        R = T_coarse[:2, :2].copy()
        tx0, ty0 = T_coarse[0, 2], T_coarse[1, 2]

        # Transform CAD points with rotation+scale only (no translation)
        cad_rotated = cad_points @ R.T  # (N, 2)

        # Image FOV bounds in world coords for filtering
        img_xmin = img_edges_world[:, 0].min()
        img_xmax = img_edges_world[:, 0].max()
        img_ymin = img_edges_world[:, 1].min()
        img_ymax = img_edges_world[:, 1].max()
        pad = 10.0  # mm margin
        fov_min = np.array([img_xmin - pad, img_ymin - pad])
        fov_max = np.array([img_xmax + pad, img_ymax + pad])

        # Pre-filter: only score CAD points that land inside the FOV
        test_shifted = cad_rotated + np.array([tx0, ty0])
        in_fov_mask = np.all(
            (test_shifted >= fov_min) & (test_shifted <= fov_max), axis=1
        )
        if in_fov_mask.sum() < 5:
            in_fov_mask = np.ones(len(cad_rotated), dtype=bool)
        cad_for_scoring = cad_rotated[in_fov_mask]

        # Subsample if too many points (speed up KDTree queries)
        if len(cad_for_scoring) > 500:
            idx = np.random.choice(len(cad_for_scoring), 500, replace=False)
            cad_for_scoring = cad_for_scoring[idx]

        # Build KDTree on image edges
        edge_tree = cKDTree(img_edges_world)

        # Score function: mean Gaussian-weighted distance
        sigma = 2.0  # mm
        sigma2 = 2.0 * sigma * sigma

        # Vectorized coarse grid search: ±100mm at 4mm steps
        coarse_offsets = np.arange(-100, 101, 4.0)
        best_score = -1.0
        best_tx, best_ty = tx0, ty0
        n_cad = len(cad_for_scoring)

        for dx in coarse_offsets:
            # Build all dy offsets at once: (n_dy, 2)
            offsets = np.column_stack([np.full(len(coarse_offsets), tx0 + dx),
                                       ty0 + coarse_offsets])
            # (n_cad, 1, 2) + (1, n_dy, 2) → (n_cad, n_dy, 2)
            shifted = cad_for_scoring[:, np.newaxis, :] + offsets[np.newaxis, :, :]
            n_dy = shifted.shape[1]
            flat = shifted.reshape(-1, 2)
            dists, _ = edge_tree.query(flat)
            dists = dists.reshape(n_cad, n_dy)
            scores = np.mean(np.exp(-dists ** 2 / sigma2), axis=0)
            best_idx = scores.argmax()
            if scores[best_idx] > best_score:
                best_score = scores[best_idx]
                best_tx = tx0 + dx
                best_ty = ty0 + coarse_offsets[best_idx]

        _print(f"  Edge translation coarse: Δ=({best_tx - tx0:+.1f}, "
               f"{best_ty - ty0:+.1f}) mm, score={best_score:.4f}")

        # Fine grid search: ±8mm at 0.5mm steps around coarse best
        fine_offsets = np.arange(-8, 8.5, 0.5)
        fine_best_score = best_score
        fine_best_tx, fine_best_ty = best_tx, best_ty

        for dx in fine_offsets:
            offsets = np.column_stack([np.full(len(fine_offsets), best_tx + dx),
                                       best_ty + fine_offsets])
            shifted = cad_for_scoring[:, np.newaxis, :] + offsets[np.newaxis, :, :]
            n_dy = shifted.shape[1]
            flat = shifted.reshape(-1, 2)
            dists, _ = edge_tree.query(flat)
            dists = dists.reshape(n_cad, n_dy)
            scores = np.mean(np.exp(-dists ** 2 / sigma2), axis=0)
            best_idx = scores.argmax()
            if scores[best_idx] > fine_best_score:
                fine_best_score = scores[best_idx]
                fine_best_tx = best_tx + dx
                fine_best_ty = best_ty + fine_offsets[best_idx]

        _print(f"  Edge translation fine: Δ=({fine_best_tx - tx0:+.2f}, "
               f"{fine_best_ty - ty0:+.2f}) mm, score={fine_best_score:.4f}")

        # Rebuild transform with refined translation
        T_refined = T_coarse.copy()
        T_refined[0, 2] = fine_best_tx
        T_refined[1, 2] = fine_best_ty

        # Compute score at original translation for comparison
        orig_dists, _ = edge_tree.query(cad_for_scoring + np.array([tx0, ty0]))
        orig_score = float(np.mean(np.exp(-orig_dists ** 2 / sigma2)))

        if fine_best_score <= orig_score * 1.001:
            _print(f"  Edge translation: no improvement, keeping original")
            return T_coarse, orig_score

        _print(f"  Edge translation: improved score {orig_score:.4f} -> {fine_best_score:.4f}")
        return T_refined, fine_best_score

    def _disambiguate_orientation(
        self,
        cad_points: np.ndarray,
        img_edges_world: np.ndarray,
        T_coarse: np.ndarray,
        pixel_size_mm: float,
        image: np.ndarray = None,
        cad_features: list = None,
    ) -> np.ndarray:
        """Resolve 180° orientation ambiguity.

        Primary method: window matching. Detected image windows have unique
        non-symmetric positions, so checking which orientation better aligns
        CAD window centers with image window centers is reliable.

        Fallback: edge overlap scoring (may fail for highly symmetric parts).
        """
        from scipy.spatial import cKDTree

        # Construct 180° candidate with corrected translation
        cad_centroid = cad_points.mean(axis=0)
        R_orig = T_coarse[:2, :2]
        t_orig = np.array([T_coarse[0, 2], T_coarse[1, 2]])
        shift = 2.0 * (R_orig @ cad_centroid)

        T_180 = T_coarse.copy()
        T_180[:2, :2] = -R_orig
        T_180[0, 2] = t_orig[0] + shift[0]
        T_180[1, 2] = t_orig[1] + shift[1]

        params_orig = affine_solver.extract_params(T_coarse)
        params_180 = affine_solver.extract_params(T_180)
        _print(f"  Orientation: orig={params_orig['rotation_deg']:.1f}°, "
               f"180°={params_180['rotation_deg']:.1f}°")

        # ── Method 1: Window-based disambiguation (fast, reliable) ────
        if image is not None and cad_features is not None:
            from .fiducial_detector import FiducialDetector
            fd = FiducialDetector()

            # Detect image windows
            image_windows = fd._window_detector.detect(image, pixel_size_mm)
            cad_window_centers = fd._extract_cad_window_centers(cad_features)

            if len(image_windows) >= 3 and len(cad_window_centers) >= 3:
                # Get image window centers in pixels
                img_win_px = np.array([
                    [w["cx"], w["cy"]] for w in image_windows
                ])

                # Score each orientation by counting how many CAD window centers
                # project near a detected image window center
                img_win_tree = cKDTree(img_win_px)

                def _window_score(T):
                    # Project CAD window centers to image pixel coords
                    projected = affine_solver.apply(T, cad_window_centers)
                    # Convert from world mm to pixels
                    px = np.column_stack([
                        projected[:, 0] / pixel_size_mm,
                        -projected[:, 1] / pixel_size_mm,
                    ])
                    # Count matches within 50px tolerance
                    dists, _ = img_win_tree.query(px)
                    matches = (dists < 50).sum()
                    return int(matches)

                score_orig = _window_score(T_coarse)
                score_180 = _window_score(T_180)

                _print(f"  Window matches: orig={score_orig}/{len(cad_window_centers)}, "
                       f"180°={score_180}/{len(cad_window_centers)}")

                if score_orig > 0 or score_180 > 0:
                    # Window matching produced results — use them
                    if score_180 > score_orig:
                        _print(f"  → 180° wins by window matching, refining...")
                        T_winner, score_final = self._refine_translation_edges(
                            cad_points, img_edges_world, T_180, pixel_size_mm,
                        )
                        p = affine_solver.extract_params(T_winner)
                        _print(f"  Selected: 180° orientation, rot={p['rotation_deg']:.2f}°")
                        return T_winner
                    else:
                        _print(f"  → Original wins by window matching, refining...")
                        T_winner, score_final = self._refine_translation_edges(
                            cad_points, img_edges_world, T_coarse, pixel_size_mm,
                        )
                        p = affine_solver.extract_params(T_winner)
                        _print(f"  Selected: original orientation, rot={p['rotation_deg']:.2f}°")
                        return T_winner
                else:
                    _print(f"  Window matching inconclusive (no matches), trying edge overlap...")

        # ── Method 2: Rotation proximity heuristic ─────────────────────
        # For near-symmetric parts, edge overlap can't distinguish: the
        # RANSAC-refined T_coarse always scores higher than T_180 (whose
        # translation is only centroid-shifted, not grid-search refined).
        # Prefer the orientation with smaller absolute rotation — the part
        # is typically placed at ~0° under the camera.
        p_orig = affine_solver.extract_params(T_coarse)
        p_180 = affine_solver.extract_params(T_180)
        rot_orig = abs(p_orig['rotation_deg']) % 360
        rot_180 = abs(p_180['rotation_deg']) % 360
        rot_orig = min(rot_orig, 360 - rot_orig)
        rot_180 = min(rot_180, 360 - rot_180)

        if rot_180 < rot_orig:
            _print(f"  → 180° candidate selected by rotation proximity "
                   f"({rot_180:.1f}° vs {rot_orig:.1f}°), refining...")
            T_winner, _ = self._refine_translation_edges(
                cad_points, img_edges_world, T_180, pixel_size_mm,
            )
            p = affine_solver.extract_params(T_winner)
            _print(f"  Selected: rot={p['rotation_deg']:.2f}°")
            return T_winner
        else:
            _print(f"  → Original selected by rotation proximity "
                   f"({rot_orig:.1f}° vs {rot_180:.1f}°), refining...")
            T_winner, _ = self._refine_translation_edges(
                cad_points, img_edges_world, T_coarse, pixel_size_mm,
            )
            p = affine_solver.extract_params(T_winner)
            _print(f"  Selected: rot={p['rotation_deg']:.2f}°")
            return T_winner

    def run_fine(
        self, ctx: RegistrationContext, coarse_transform: np.ndarray,
    ) -> FineResult:
        _print("-" * 60)
        _print("REFINEMENT (Fiducial ICP)")

        group, features = _resolve_features(ctx)
        if not features:
            return FineResult(transform=coarse_transform, error=float("inf"))

        cad_pts = self._silhouette_gen.generate_point_cloud(features, density=0.5)
        if len(cad_pts) < 3:
            return FineResult(transform=coarse_transform, error=float("inf"))

        coarse_data = ctx.debug_data.get("coarse", {})
        img_edges_world = coarse_data.get("img_edges_world")
        if img_edges_world is None or len(img_edges_world) < 3:
            img_edges_world = coarse_data.get("img_contour_world")
        if img_edges_world is None or len(img_edges_world) < 3:
            _print("  ERROR: no cached image data")
            return FineResult(transform=coarse_transform, error=float("inf"))

        _print(f"  CAD: {len(cad_pts)} pts, Image: {len(img_edges_world)} pts")

        result = self._refinement.refine(cad_pts, img_edges_world, coarse_transform)
        T_refined = result["transform"]

        img_eval = img_edges_world if len(img_edges_world) >= 10 else coarse_data.get("img_contour_world")
        if img_eval is None or len(img_eval) < 3:
            return FineResult(transform=coarse_transform, error=float("inf"))

        coarse_rmse = _compute_image_rmse(img_eval, cad_pts, coarse_transform)
        refined_rmse = _compute_image_rmse(img_eval, cad_pts, T_refined)

        if refined_rmse < coarse_rmse:
            # Verify the refined transform hasn't drifted too far from
            # the fiducial-based coarse transform. ICP can converge to
            # wrong local minima for grid-like structures.
            coarse_params = affine_solver.extract_params(coarse_transform)
            refined_params = affine_solver.extract_params(T_refined)
            scale_drift = abs(refined_params['scale_x'] - coarse_params['scale_x'])
            rot_drift = abs(refined_params['rotation_deg'] - coarse_params['rotation_deg'])
            if scale_drift > 0.02 or rot_drift > 2.0:
                _print(f"  ICP drifted too much (scale Δ={scale_drift:.4f}, "
                       f"rot Δ={rot_drift:.2f}°), keeping fiducial transform")
                T_final = coarse_transform
                refined_rmse = coarse_rmse
            else:
                T_final = T_refined
                _print(f"  Refined ({result['iterations']} iters): "
                       f"RMSE {coarse_rmse:.4f} -> {refined_rmse:.4f} mm (improved)")
        else:
            T_final = coarse_transform
            _print(f"  ICP did not improve ({coarse_rmse:.4f} -> {refined_rmse:.4f} mm), "
                   f"keeping fiducial transform")

        final_mse = min(coarse_rmse, refined_rmse) ** 2

        ctx.debug_data["fine"] = {
            "transform": T_final,
            "iterations": result["iterations"],
            "error": final_mse,
            "converged": refined_rmse < coarse_rmse,
            "cad_contour": cad_pts,
            "img_world": img_edges_world,
            "strategy": "fiducial",
        }

        return FineResult(
            transform=T_final,
            error=final_mse,
            iterations=result["iterations"],
            converged=refined_rmse < coarse_rmse,
        )


class TeachICPStrategy(RegistrationStrategy):
    """Pose template + constrained ICP refinement.

    The user teaches two CAD-to-image point correspondences. The resulting
    similarity transform is saved as a JSON template. At runtime the
    template is loaded and ICP performs only local refinement within
    tight bounds (±5mm translation, ±2° rotation, ±1% scale).
    """

    def __init__(self) -> None:
        from .constrained_icp import ConstrainedICP
        self._refinement = ConstrainedICP(
            max_iterations=30,
            tolerance=1e-4,
            outlier_distance=15.0,
            max_translation=10.0,
            max_rotation_deg=3.0,
            max_scale_change=0.02,
        )
        self._silhouette_gen = RegistrationContourGenerator()

    @property
    def name(self) -> str:
        return "Teach + ICP"

    @property
    def description(self) -> str:
        return "Pose template + constrained refinement"

    @staticmethod
    def _pose_template_path(ctx_or_info) -> str:
        import os
        # Accept either a RegistrationContext or a dict with keys
        if hasattr(ctx_or_info, 'image_path'):
            image_path = ctx_or_info.image_path
            group_id = ctx_or_info.group_id
        else:
            image_path = ctx_or_info.get("image_path", "")
            group_id = ctx_or_info.get("group_id", "default")
        if not image_path:
            return ""
        return os.path.join(os.path.dirname(image_path), f"{group_id}_pose.json")

    @staticmethod
    def _load_pose_template(path: str) -> dict | None:
        import json, os
        if not path or not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _save_pose_template(path: str, data: dict) -> None:
        import json, os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _compute_transform_from_points(
        cad_points: list, img_points: list, pixel_size_mm: float,
    ) -> np.ndarray:
        """Compute similarity transform from 2 CAD↔Image point pairs."""
        ps = pixel_size_mm
        c1 = np.array(cad_points[0]["world"], dtype=np.float64)
        c2 = np.array(cad_points[1]["world"], dtype=np.float64)
        p1 = np.array([img_points[0]["pixel"][0] * ps,
                        -img_points[0]["pixel"][1] * ps], dtype=np.float64)
        p2 = np.array([img_points[1]["pixel"][0] * ps,
                        -img_points[1]["pixel"][1] * ps], dtype=np.float64)
        return affine_solver.solve_similarity(
            np.array([c1, c2]),
            np.array([p1, p2]),
        )

    @staticmethod
    def _template_to_transform(template: dict) -> np.ndarray:
        import math
        rot = math.radians(template["rotation_deg"])
        scale = template["scale"]
        tx, ty = template["translation"]
        T = np.eye(3, dtype=np.float64)
        T[0, 0] = scale * math.cos(rot)
        T[0, 1] = -scale * math.sin(rot)
        T[1, 0] = scale * math.sin(rot)
        T[1, 1] = scale * math.cos(rot)
        T[0, 2] = tx
        T[1, 2] = ty
        return T

    @staticmethod
    def _compute_pixel_to_world_transform(
        registration_transform: np.ndarray,
        pixel_size_mm: float,
    ) -> np.ndarray:
        """Compute pixel → CAD/world transform from CAD → image-world registration.

        This mirrors RegistrationPanel._compute_image_affine so local fitting
        projects CAD features exactly as the canvas/measurement code does.
        """
        T_pixel_to_imgworld = np.array([
            [pixel_size_mm,  0,  0],
            [0, -pixel_size_mm,  0],
            [0,  0,  1],
        ], dtype=np.float64)
        return np.linalg.inv(registration_transform) @ T_pixel_to_imgworld

    @staticmethod
    def _local_rmse(
        src_points: np.ndarray,
        dst_points: np.ndarray,
        transform: np.ndarray,
    ) -> float:
        if len(src_points) == 0 or len(dst_points) == 0:
            return float("inf")
        transformed = affine_solver.apply(transform, src_points)
        dists = np.sqrt(np.sum((transformed - dst_points) ** 2, axis=1))
        return float(np.sqrt(np.mean(dists ** 2)))

    @staticmethod
    def _clamp_transform_update(
        candidate: np.ndarray,
        reference: np.ndarray,
        max_translation: float = 10.0,
        max_rotation_deg: float = 3.0,
    ) -> tuple[np.ndarray, bool]:
        """Clamp a candidate similarity transform around the taught pose."""
        ref_params = affine_solver.extract_params(reference)
        cand_params = affine_solver.extract_params(candidate)
        ref_scale = ref_params["scale_x"]
        ref_rot = ref_params["rotation_deg"]
        cand_rot = cand_params["rotation_deg"]

        rot_delta = cand_rot - ref_rot
        while rot_delta > 180:
            rot_delta -= 360
        while rot_delta < -180:
            rot_delta += 360

        clamped = False
        if abs(rot_delta) > max_rotation_deg:
            rot_delta = np.sign(rot_delta) * max_rotation_deg
            clamped = True

        tx = cand_params["tx"]
        ty = cand_params["ty"]
        dx = tx - ref_params["tx"]
        dy = ty - ref_params["ty"]
        dist = float(np.sqrt(dx * dx + dy * dy))
        if dist > max_translation and dist > 1e-12:
            factor = max_translation / dist
            tx = ref_params["tx"] + dx * factor
            ty = ref_params["ty"] + dy * factor
            clamped = True

        rot = np.radians(ref_rot + rot_delta)
        T = np.eye(3, dtype=np.float64)
        T[0, 0] = ref_scale * np.cos(rot)
        T[0, 1] = -ref_scale * np.sin(rot)
        T[1, 0] = ref_scale * np.sin(rot)
        T[1, 1] = ref_scale * np.cos(rot)
        T[0, 2] = tx
        T[1, 2] = ty
        return T, clamped

    def _fit_selected_lines(
        self,
        features: list[CADFeature],
        image: np.ndarray,
        registration_transform: np.ndarray,
        pixel_size_mm: float,
    ) -> dict:
        """Fit selected CAD LINE features against local image edges."""
        if not HAS_CV2:
            return {"success": False, "reason": "cv2 unavailable"}

        line_features = [f for f in features if f.feature_type == FeatureType.LINE]
        if len(line_features) < 2:
            return {"success": False, "reason": "not enough line features"}

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)

        from ..measurement.line_fitter import LineFittingEngine
        fitter = LineFittingEngine(gradient)

        pixel_to_world = self._compute_pixel_to_world_transform(
            registration_transform, pixel_size_mm,
        )
        world_to_pixel = np.linalg.inv(pixel_to_world)
        img_h, img_w = gray.shape[:2]

        cad_src: list[np.ndarray] = []
        img_dst: list[np.ndarray] = []
        fitted_edge_points: list[np.ndarray] = []
        predicted_edge_points: list[np.ndarray] = []
        accepted = 0
        rejected = 0
        fit_debug = []

        for feat in line_features:
            geom = feat.geometry
            cad_p1 = np.array([geom["x1"], geom["y1"]], dtype=np.float64)
            cad_p2 = np.array([geom["x2"], geom["y2"]], dtype=np.float64)
            cad_vec = cad_p2 - cad_p1
            cad_len2 = float(np.dot(cad_vec, cad_vec))
            if cad_len2 < 1e-9:
                rejected += 1
                continue

            pred_px = affine_solver.apply(world_to_pixel, np.array([cad_p1, cad_p2]))
            pred_p1, pred_p2 = pred_px[0], pred_px[1]
            px_vec = pred_p2 - pred_p1
            px_len = float(np.linalg.norm(px_vec))
            if px_len < 5.0:
                rejected += 1
                continue

            # Keep lines that are at least partly in/near the image.
            margin = max(80.0, px_len * 0.20)
            if ((max(pred_p1[0], pred_p2[0]) < -margin)
                    or (min(pred_p1[0], pred_p2[0]) > img_w + margin)
                    or (max(pred_p1[1], pred_p2[1]) < -margin)
                    or (min(pred_p1[1], pred_p2[1]) > img_h + margin)):
                rejected += 1
                continue

            result = fitter.fit(
                pred_p1, pred_p2,
                n_scanlines=80,
                scan_width=max(25.0, px_len * 0.08),
                min_gradient=15.0,
            )
            if result is None or result.confidence < 0.20 or result.n_edge_points < 8:
                rejected += 1
                continue

            edge_points = result.edge_points
            if len(edge_points) == 0:
                rejected += 1
                continue

            # Reject fits that moved implausibly far from the predicted line.
            pred_dir = px_vec / px_len
            pred_normal = np.array([-pred_dir[1], pred_dir[0]])
            perp_offsets = np.abs((edge_points - pred_p1) @ pred_normal)
            max_allowed_offset = max(60.0, px_len * 0.18)
            if float(np.median(perp_offsets)) > max_allowed_offset:
                rejected += 1
                continue

            # Pair each fitted image edge point with the corresponding CAD
            # point at the same projected line parameter.
            t = ((edge_points - pred_p1) @ pred_dir) / px_len
            valid = (t >= -0.10) & (t <= 1.10)
            if valid.sum() < 8:
                rejected += 1
                continue
            t = np.clip(t[valid], 0.0, 1.0)
            edge_points = edge_points[valid]

            cad_points = cad_p1 + np.outer(t, cad_vec)
            img_world = np.column_stack([
                edge_points[:, 0] * pixel_size_mm,
                -edge_points[:, 1] * pixel_size_mm,
            ])
            pred_world = np.column_stack([
                (pred_p1[0] + t * px_vec[0]) * pixel_size_mm,
                -(pred_p1[1] + t * px_vec[1]) * pixel_size_mm,
            ])

            cad_src.append(cad_points)
            img_dst.append(img_world)
            fitted_edge_points.append(img_world)
            predicted_edge_points.append(pred_world)
            accepted += 1
            fit_debug.append({
                "feature_id": feat.feature_id,
                "n_edge_points": int(len(edge_points)),
                "confidence": float(result.confidence),
                "residual_px": float(result.residual),
                "median_offset_px": float(np.median(perp_offsets)),
            })

        if not cad_src or not img_dst:
            return {
                "success": False,
                "reason": "no local line fits accepted",
                "accepted": accepted,
                "rejected": rejected,
            }

        return {
            "success": True,
            "cad_points": np.vstack(cad_src),
            "image_world_points": np.vstack(img_dst),
            "fitted_edge_points": np.vstack(fitted_edge_points),
            "predicted_edge_points": np.vstack(predicted_edge_points),
            "accepted": accepted,
            "rejected": rejected,
            "fit_debug": fit_debug,
        }

    def _refine_from_line_fits(
        self,
        features: list[CADFeature],
        image: np.ndarray,
        initial_transform: np.ndarray,
        pixel_size_mm: float,
    ) -> tuple[np.ndarray, dict]:
        """Refine a taught pose using local fits of selected CAD lines."""
        fit = self._fit_selected_lines(
            features, image, initial_transform, pixel_size_mm,
        )
        if not fit.get("success"):
            return initial_transform, fit

        cad_points = fit["cad_points"]
        image_world = fit["image_world_points"]
        if len(cad_points) < 12:
            fit["success"] = False
            fit["reason"] = "too few fitted edge points"
            return initial_transform, fit

        before_rmse = self._local_rmse(cad_points, image_world, initial_transform)
        ref_scale = affine_solver.extract_params(initial_transform)["scale_x"]
        candidate = affine_solver.solve_rigid_with_fixed_scale(
            cad_points, image_world, ref_scale,
        )
        candidate, clamped = self._clamp_transform_update(
            candidate, initial_transform,
            max_translation=10.0,
            max_rotation_deg=3.0,
        )
        after_rmse = self._local_rmse(cad_points, image_world, candidate)

        fit["before_rmse"] = before_rmse
        fit["after_rmse"] = after_rmse
        fit["clamped"] = clamped

        if not np.isfinite(after_rmse) or after_rmse >= before_rmse * 0.995:
            fit["success"] = False
            fit["reason"] = "local line fit did not improve RMSE"
            return initial_transform, fit

        fit["success"] = True
        return candidate, fit

    def _refine_translation(
        self,
        cad_points: np.ndarray,
        img_edges_world: np.ndarray,
        T_template: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Grid-search for best translation (rotation+scale fixed).

        The 2-point teach template often has large translation error
        (10-50mm) but reasonable rotation and scale.  This method keeps
        the linear part of T_template and searches over a translation
        grid to maximise Gaussian-weighted edge overlap.

        Returns (T_refined, score).
        """
        from scipy.spatial import cKDTree

        if len(cad_points) < 10 or len(img_edges_world) < 10:
            return T_template, 0.0

        R = T_template[:2, :2].copy()
        tx0, ty0 = T_template[0, 2], T_template[1, 2]

        cad_rotated = cad_points @ R.T

        pad = 10.0
        fov_min = np.array([img_edges_world[:, 0].min() - pad,
                            img_edges_world[:, 1].min() - pad])
        fov_max = np.array([img_edges_world[:, 0].max() + pad,
                            img_edges_world[:, 1].max() + pad])

        test = cad_rotated + np.array([tx0, ty0])
        in_fov = np.all((test >= fov_min) & (test <= fov_max), axis=1)
        if in_fov.sum() < 5:
            in_fov = np.ones(len(cad_rotated), dtype=bool)
        cad_scoring = cad_rotated[in_fov]

        if len(cad_scoring) > 500:
            idx = np.random.choice(len(cad_scoring), 500, replace=False)
            cad_scoring = cad_scoring[idx]

        edge_tree = cKDTree(img_edges_world)
        sigma = 2.0
        sigma2 = 2.0 * sigma * sigma

        # Coarse grid: ±200mm at 5mm steps
        coarse = np.arange(-200, 201, 5.0)
        n_cad = len(cad_scoring)
        best_score = -1.0
        best_tx, best_ty = tx0, ty0

        for dx in coarse:
            offsets = np.column_stack([np.full(len(coarse), tx0 + dx),
                                       ty0 + coarse])
            shifted = cad_scoring[:, np.newaxis, :] + offsets[np.newaxis, :, :]
            flat = shifted.reshape(-1, 2)
            dists, _ = edge_tree.query(flat)
            dists = dists.reshape(n_cad, len(coarse))
            scores = np.mean(np.exp(-dists ** 2 / sigma2), axis=0)
            bi = scores.argmax()
            if scores[bi] > best_score:
                best_score = scores[bi]
                best_tx = tx0 + dx
                best_ty = ty0 + coarse[bi]

        _print(f"  Translation coarse: ({tx0:.1f}, {ty0:.1f}) -> "
               f"({best_tx:.1f}, {best_ty:.1f})  score={best_score:.4f}")

        # Fine grid: ±10mm at 0.5mm steps
        fine = np.arange(-10, 10.5, 0.5)
        for dx in fine:
            offsets = np.column_stack([np.full(len(fine), best_tx + dx),
                                       best_ty + fine])
            shifted = cad_scoring[:, np.newaxis, :] + offsets[np.newaxis, :, :]
            flat = shifted.reshape(-1, 2)
            dists, _ = edge_tree.query(flat)
            dists = dists.reshape(n_cad, len(fine))
            scores = np.mean(np.exp(-dists ** 2 / sigma2), axis=0)
            bi = scores.argmax()
            if scores[bi] > best_score:
                best_score = scores[bi]
                best_tx = best_tx + dx
                best_ty = best_ty + fine[bi]

        _print(f"  Translation fine:   ({best_tx:.1f}, {best_ty:.1f})  "
               f"score={best_score:.4f}")

        T_refined = T_template.copy()
        T_refined[0, 2] = best_tx
        T_refined[1, 2] = best_ty
        return T_refined, best_score

    def run_coarse(self, ctx: RegistrationContext) -> CoarseResult:
        _print("=" * 60)
        _print("COARSE REGISTRATION (Teach + ICP)")
        _print(f"  pixel_size_mm = {ctx.pixel_size_mm}")

        group, features = _resolve_features(ctx)
        if not features:
            _print("  ERROR: no features available")
            return CoarseResult(transform=affine_solver.identity(), error=float("inf"))

        _print(f"  Group: {group.name if group else 'all features'} ({len(features)} features)")

        path = self._pose_template_path(ctx)
        template = self._load_pose_template(path)
        if template is None:
            _print(f"  ERROR: no pose template found at {path}")
            _print(f"  Use 'Teach Initial Pose' to create one first.")
            return CoarseResult(transform=affine_solver.identity(), error=float("inf"))

        T = self._template_to_transform(template)
        params = affine_solver.extract_params(T)
        _print(f"  Template: scale={params['scale_x']:.6f}, "
               f"rot={params['rotation_deg']:.2f}°, "
               f"tx={params['tx']:.2f}, ty={params['ty']:.2f}")

        from .image_extractor import ImageFeatureExtractor
        try:
            image = ImageFeatureExtractor.load_image(ctx.image_path)
            cad_points = self._silhouette_gen.generate_point_cloud(features, density=0.5)
            img_edges = ImageFeatureExtractor.extract_edges(image)
            img_edges_world = img_edges.astype(np.float64)
            img_edges_world[:, 0] *= ctx.pixel_size_mm
            img_edges_world[:, 1] *= -ctx.pixel_size_mm

            error = _compute_image_rmse(img_edges_world, cad_points, T)
            _print(f"  Template RMSE: {error:.4f} mm")

            ctx.debug_data["coarse"] = {
                "cad_points": cad_points,
                "cad_centroid": cad_points.mean(axis=0) if len(cad_points) > 0 else np.zeros(2),
                "image_edges": img_edges,
                "img_edges_world": img_edges_world,
                "img_contour_world": img_edges_world,
                "transform": T,
                "pixel_size_mm": ctx.pixel_size_mm,
                "image_path": ctx.image_path,
                "strategy": "teach_icp",
            }
        except Exception as e:
            _print(f"  Warning: image extraction failed: {e}")
            error = 0.0

        return CoarseResult(transform=T, error=error)

    def run_fine(
        self, ctx: RegistrationContext, coarse_transform: np.ndarray,
    ) -> FineResult:
        _print("-" * 60)
        _print("REFINEMENT (Selected Line Edge Fit + Constrained ICP fallback)")

        group, features = _resolve_features(ctx)
        if not features:
            return FineResult(transform=coarse_transform, error=float("inf"))

        selected_features = features
        if group and group.feature_ids:
            selected_features = [
                f for f in (ctx.repo.get(fid) for fid in group.feature_ids)
                if f is not None
            ]

        cad_pts = self._silhouette_gen.generate_point_cloud(features, density=0.5)
        if len(cad_pts) < 3:
            return FineResult(transform=coarse_transform, error=float("inf"))

        coarse_data = ctx.debug_data.get("coarse", {})
        img_edges_world = coarse_data.get("img_edges_world")
        if img_edges_world is None or len(img_edges_world) < 3:
            img_edges_world = coarse_data.get("img_contour_world")
        if img_edges_world is None or len(img_edges_world) < 3:
            _print("  ERROR: no cached image data")
            return FineResult(transform=coarse_transform, error=float("inf"))

        _print(f"  CAD: {len(cad_pts)} pts, Image: {len(img_edges_world)} pts")

        template_rmse = _compute_image_rmse(img_edges_world, cad_pts, coarse_transform)
        _print(f"  Template RMSE: {template_rmse:.4f} mm")

        image_path = ctx.image_path or coarse_data.get("image_path", "")
        line_fit = {"success": False, "reason": "image path unavailable"}
        T_line = coarse_transform
        if image_path:
            try:
                image = ImageFeatureExtractor.load_image(image_path)
                T_line, line_fit = self._refine_from_line_fits(
                    selected_features, image, coarse_transform, ctx.pixel_size_mm,
                )
            except Exception as e:
                line_fit = {"success": False, "reason": f"line fit failed: {e}"}

        if line_fit.get("success"):
            line_rmse = _compute_image_rmse(img_edges_world, cad_pts, T_line)
            _print(
                "  Local line fit: "
                f"{line_fit.get('accepted', 0)} accepted, "
                f"{line_fit.get('rejected', 0)} rejected"
            )
            _print(
                "  Local line RMSE: "
                f"{line_fit['before_rmse']:.4f} -> "
                f"{line_fit['after_rmse']:.4f} mm "
                f"(global edge RMSE {line_rmse:.4f} mm)"
            )
            if line_fit.get("clamped"):
                _print("  Local line update was clamped to the teach-pose bounds")

            T_final = T_line
            final_rmse = line_rmse

            ctx.debug_data["fine"] = {
                "transform": T_final,
                "iterations": 1,
                "error": final_rmse ** 2,
                "converged": True,
                "cad_contour": cad_pts,
                "img_world": img_edges_world,
                "strategy": "teach_icp",
                "line_fit": line_fit,
                "delta_translation": 0.0,
                "delta_rotation": 0.0,
            }

            if "coarse" in ctx.debug_data:
                ctx.debug_data["coarse"]["transform"] = T_final

            return FineResult(
                transform=T_final,
                error=final_rmse ** 2,
                iterations=1,
                converged=True,
            )

        _print(f"  Local line fit unavailable: {line_fit.get('reason', 'unknown')}")
        _print("  Falling back to translation search + constrained ICP")

        # Fallback: translation grid search for legacy/non-line groups.
        T_aligned, _ = self._refine_translation(
            cad_pts, img_edges_world, coarse_transform,
        )
        aligned_rmse = _compute_image_rmse(img_edges_world, cad_pts, T_aligned)
        _print(f"  After translation search: RMSE {template_rmse:.4f} -> "
               f"{aligned_rmse:.4f} mm")
        if aligned_rmse < template_rmse:
            icp_start = T_aligned
        else:
            _print("  Translation search did not improve, using template")
            icp_start = coarse_transform

        # Constrained ICP fallback. This is intentionally not used when
        # selected-line fitting succeeds, because repeated window grids can
        # pull nearest-neighbor ICP to a wrong but locally dense edge set.
        result = self._refinement.refine(cad_pts, img_edges_world, icp_start)
        T_refined = result["transform"]

        ref_params = affine_solver.extract_params(icp_start)
        new_params = affine_solver.extract_params(T_refined)
        dt = np.sqrt((new_params["tx"] - ref_params["tx"]) ** 2 +
                       (new_params["ty"] - ref_params["ty"]) ** 2)
        drot = abs(new_params["rotation_deg"] - ref_params["rotation_deg"])
        if drot > 180:
            drot = 360 - drot

        img_eval = img_edges_world if len(img_edges_world) >= 10 else coarse_data.get("img_contour_world")
        if img_eval is not None and len(img_eval) >= 3:
            icp_start_rmse = _compute_image_rmse(img_eval, cad_pts, icp_start)
            refined_rmse = _compute_image_rmse(img_eval, cad_pts, T_refined)
        else:
            icp_start_rmse = float("inf")
            refined_rmse = float("inf")

        _print(f"  ICP: {result['iterations']} iters, "
               f"dt={dt:.3f}mm, drot={drot:.2f}deg, "
               f"clamped={result.get('clamped', False)}")
        _print(f"  RMSE: {icp_start_rmse:.4f} -> {refined_rmse:.4f} mm")

        use_refined = refined_rmse < icp_start_rmse
        T_final = T_refined if use_refined else icp_start

        final_rmse = min(icp_start_rmse, refined_rmse)

        ctx.debug_data["fine"] = {
            "transform": T_final,
            "iterations": result["iterations"],
            "error": final_rmse ** 2,
            "converged": use_refined and result.get("converged", False),
            "cad_contour": cad_pts,
            "img_world": img_edges_world,
            "strategy": "teach_icp",
            "line_fit": line_fit,
            "delta_translation": dt,
            "delta_rotation": drot,
        }

        if "coarse" in ctx.debug_data:
            ctx.debug_data["coarse"]["transform"] = T_final

        return FineResult(
            transform=T_final,
            error=final_rmse ** 2,
            iterations=result["iterations"],
            converged=use_refined and result.get("converged", False),
        )

STRATEGY_REGISTRY: dict[str, type[RegistrationStrategy]] = {
    "full_silhouette": FullSilhouetteStrategy,
    "convex_hull": ConvexHullStrategy,
    "fiducial": FiducialStrategy,
    "teach_icp": TeachICPStrategy,
}

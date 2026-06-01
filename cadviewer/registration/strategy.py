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
from . import affine_solver

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
        return None, []
    features = [ctx.repo.get(fid) for fid in group.feature_ids]
    return group, [f for f in features if f is not None]


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
        if not group or not features:
            _print("  ERROR: empty group")
            return CoarseResult(transform=affine_solver.identity(), error=float("inf"))

        sil_types = {"LINE", "POLYLINE", "ARC"}
        sil_count = sum(1 for f in features if f.feature_type.name in sil_types)
        _print(f"  Group: {group.name} ({len(features)} features, {sil_count} silhouette)")

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
        if not group or not features:
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
        if not group or not features:
            _print("  ERROR: empty group")
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
        # Use edge points (reliable) instead of silhouette contour
        # (may be broken for bright-on-light telecentric images).
        error = _compute_image_rmse(img_edges_world, cad_points, T_coarse)
        _print(f"  Image→CAD RMSE: {error:.4f} mm")
        _print(f"  Aligner inlier fraction: {rect_info.get('score', 'N/A')}")

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
        if not group or not features:
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
            _print(f"  Refined ({result['iterations']} iters): "
                   f"RMSE {coarse_rmse:.4f} → {refined_rmse:.4f} mm (improved)")
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


# ── Strategy registry ────────────────────────────────────────────────

STRATEGY_REGISTRY: dict[str, type[RegistrationStrategy]] = {
    "full_silhouette": FullSilhouetteStrategy,
    "convex_hull": ConvexHullStrategy,
}

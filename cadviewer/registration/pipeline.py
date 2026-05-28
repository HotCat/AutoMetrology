"""
RegistrationPipeline — silhouette-based CAD-to-image alignment.

Architecture:
  1. Coarse: CAD silhouette + image silhouette → minAreaRect alignment
  2. Refinement: Optional lightweight outer-contour ICP
  3. Freeze: Global transform is locked after alignment
  4. Metrology: Local ROI subpixel fitting (separate from registration)

Registration uses only simple global silhouette geometry.
Precise measurement happens via local subpixel fitting AFTER registration.

This separation prevents ICP local minima caused by repetitive internal
CAD geometry (circles, nested contours, parallel lines).
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Optional

from ..models.repository import FeatureRepository
from ..models.registration import RegistrationManager
from .cad_silhouette import CADSilhouetteExtractor, RegistrationContourGenerator
from .image_silhouette import ProductSilhouetteExtractor
from .min_area_rect_reg import MinAreaRectRegistration
from .contour_refinement import ContourRefinementEngine
from .image_extractor import ImageFeatureExtractor
from . import affine_solver

logger = logging.getLogger(__name__)


def _print(msg: str) -> None:
    print(f"[REG] {msg}")


class RegistrationPipeline:
    """Silhouette-based registration pipeline.

    Separates global registration (silhouette alignment) from
    local metrology (subpixel feature fitting).
    """

    def __init__(
        self,
        repo: FeatureRepository,
        reg_manager: RegistrationManager,
    ) -> None:
        self._repo = repo
        self._reg_manager = reg_manager
        self._silhouette_gen = RegistrationContourGenerator()
        self._img_silhouette = ProductSilhouetteExtractor()
        self._min_area_rect = MinAreaRectRegistration()
        self._refinement = ContourRefinementEngine(
            max_iterations=30, tolerance=1e-4, outlier_distance=5.0,
        )
        self._debug_data: dict = {}

    def run_coarse(
        self,
        image_path: str,
        group_id: str,
        pixel_size_mm: float,
    ) -> dict:
        """Coarse registration: silhouette minAreaRect alignment.

        Args:
            image_path: path to telecentric image
            group_id: registration group to use
            pixel_size_mm: mm per pixel

        Returns dict with 'transform', 'stage', 'error'.
        """
        _print("=" * 60)
        _print("COARSE REGISTRATION (MinAreaRect Silhouette)")
        _print(f"  pixel_size_mm = {pixel_size_mm}")

        group = self._reg_manager.get_group(group_id)
        if not group or not group.feature_ids:
            _print("  ERROR: empty group")
            return {
                "transform": affine_solver.identity(),
                "stage": "coarse",
                "error": float("inf"),
            }

        features = [self._repo.get(fid) for fid in group.feature_ids]
        features = [f for f in features if f is not None]

        # ── Step 1: Extract CAD silhouette ────────────────────────
        _print(f"  Group: {group.name} ({len(features)} features)")
        silhouette_types = {"LINE", "POLYLINE", "ARC"}
        sil_count = sum(1 for f in features if f.feature_type.name in silhouette_types)
        _print(f"  Silhouette-relevant features: {sil_count}/{len(features)}")

        cad_points = self._silhouette_gen.generate_point_cloud(
            features, density=0.5,
        )
        cad_contour = self._silhouette_gen.generate(features, density=0.5)

        if len(cad_points) < 3:
            _print("  ERROR: too few CAD silhouette points")
            return {
                "transform": affine_solver.identity(),
                "stage": "coarse",
                "error": float("inf"),
            }

        cad_centroid = cad_points.mean(axis=0)
        _print(f"  CAD silhouette: {len(cad_points)} points, "
               f"contour: {len(cad_contour) if cad_contour is not None else 0} pts")
        _print(f"    centroid: ({cad_centroid[0]:.3f}, {cad_centroid[1]:.3f})")

        # ── Step 2: Extract image silhouette ──────────────────────
        image = ImageFeatureExtractor.load_image(image_path)
        _print(f"  Image: {image.shape[1]}x{image.shape[0]} pixels")

        mask, img_contour = self._img_silhouette.extract(image)
        if len(img_contour) < 3:
            _print("  ERROR: too few image silhouette points")
            return {
                "transform": affine_solver.identity(),
                "stage": "coarse",
                "error": float("inf"),
            }

        _print(f"  Image silhouette: {len(img_contour)} contour points")

        # ── Step 3: MinAreaRect registration ──────────────────────
        T_coarse, rect_info = self._min_area_rect.register(
            cad_points, img_contour, pixel_size_mm,
        )

        if np.allclose(T_coarse, np.eye(3)):
            _print("  ERROR: minAreaRect registration failed")
            return {
                "transform": affine_solver.identity(),
                "stage": "coarse",
                "error": float("inf"),
            }

        params = affine_solver.extract_params(T_coarse)
        _print(f"  MinAreaRect transform:")
        _print(f"    scale={params['scale_x']:.6f}  "
               f"rotation={params['rotation_deg']:.4f}deg")
        _print(f"    tx={params['tx']:.4f}  ty={params['ty']:.4f}")

        # Compute RMSE
        img_world = img_contour.copy().astype(np.float64)
        img_world[:, 0] *= pixel_size_mm
        img_world[:, 1] *= -pixel_size_mm

        error = self._compute_rmse(cad_points, img_world, T_coarse)
        _print(f"  Coarse RMSE: {error:.4f} mm")

        # Also extract full image edges for debug overlay
        image_edges = ImageFeatureExtractor.extract_edges(image)

        # ── Store debug data ──────────────────────────────────────
        self._debug_data["coarse"] = {
            "cad_points": cad_points,
            "cad_contour": cad_contour,
            "image_edges": image_edges,
            "img_contour": img_contour,
            "img_contour_world": img_world,
            "mask": mask,
            "transform": T_coarse,
            "rect_info": rect_info,
            "pixel_size_mm": pixel_size_mm,
            "cad_centroid": cad_centroid,
            "image_path": image_path,
        }

        return {"transform": T_coarse, "stage": "coarse", "error": error}

    def run_fine(
        self,
        coarse_transform: np.ndarray,
        group_id: str,
        pixel_size_mm: Optional[float] = None,
    ) -> dict:
        """Fine registration: lightweight outer contour refinement.

        Uses only the CAD silhouette contour — no internal features.
        """
        _print("-" * 60)
        _print("REFINEMENT (Outer Contour ICP)")

        coarse_params = affine_solver.extract_params(coarse_transform)
        _print(f"  Input transform:")
        _print(f"    scale={coarse_params['scale_x']:.6f}  "
               f"rotation={coarse_params['rotation_deg']:.4f}deg")

        group = self._reg_manager.get_group(group_id)
        if not group or not group.feature_ids:
            return {
                "transform": coarse_transform, "stage": "fine",
                "iterations": 0, "error": float("inf"), "converged": False,
            }

        features = [self._repo.get(fid) for fid in group.feature_ids]
        features = [f for f in features if f is not None]

        # Use only CAD silhouette contour for refinement (NOT all features)
        cad_contour = self._silhouette_gen.generate(features, density=0.5)
        if cad_contour is None or len(cad_contour) < 3:
            _print("  ERROR: no CAD silhouette contour for refinement")
            return {
                "transform": coarse_transform, "stage": "fine",
                "iterations": 0, "error": float("inf"), "converged": False,
            }

        _print(f"  CAD contour: {len(cad_contour)} points (silhouette only)")

        coarse_data = self._debug_data.get("coarse", {})
        img_world = coarse_data.get("img_contour_world")
        if img_world is None or len(img_world) < 3:
            _print("  ERROR: no cached image silhouette from coarse stage")
            return {
                "transform": coarse_transform, "stage": "fine",
                "iterations": 0, "error": float("inf"), "converged": False,
            }

        _print(f"  Image silhouette: {len(img_world)} points")

        # Run lightweight contour refinement (ICP on silhouette only)
        result = self._refinement.refine(
            cad_contour, img_world, coarse_transform,
        )
        T_refined = result["transform"]

        refined_params = affine_solver.extract_params(T_refined)
        _print(f"  Refined ({result['iterations']} iters):")
        _print(f"    scale={refined_params['scale_x']:.6f}  "
               f"rotation={refined_params['rotation_deg']:.4f}deg")
        _print(f"    scale delta = "
               f"{refined_params['scale_x'] - coarse_params['scale_x']:.6f}")
        _print(f"    rotation delta = "
               f"{refined_params['rotation_deg'] - coarse_params['rotation_deg']:.4f}deg")
        _print(f"    RMSE = {np.sqrt(result['final_error']):.4f} mm  "
               f"converged={result['converged']}")

        # Log image affine preview
        ps = coarse_data.get("pixel_size_mm", pixel_size_mm or 0.01)
        T_pixel_to_imgworld = np.array([
            [ps, 0, 0], [0, -ps, 0], [0, 0, 1],
        ], dtype=np.float64)
        T_imgworld_to_cad = np.linalg.inv(T_refined)
        T_img = T_imgworld_to_cad @ T_pixel_to_imgworld
        img_params = affine_solver.extract_params(T_img)
        _print(f"  Image affine (pixel → CAD):")
        _print(f"    scale={img_params['scale_x']:.6f}  "
               f"rotation={img_params['rotation_deg']:.4f}deg  "
               f"tx={img_params['tx']:.4f}  ty={img_params['ty']:.4f}")

        self._debug_data["fine"] = {
            "transform": T_refined,
            "iterations": result["iterations"],
            "error": result["final_error"],
            "converged": result["converged"],
            "cad_contour": cad_contour,
            "img_world": img_world,
        }

        return {
            "transform": T_refined,
            "stage": "fine",
            "iterations": result["iterations"],
            "error": result["final_error"],
            "converged": result["converged"],
        }

    def run_full(
        self,
        image_path: str,
        group_id: str,
        pixel_size_mm: float,
    ) -> dict:
        """Run complete coarse → refinement pipeline."""
        coarse = self.run_coarse(image_path, group_id, pixel_size_mm)
        if coarse["error"] == float("inf"):
            return coarse

        fine = self.run_fine(coarse["transform"], group_id, pixel_size_mm)

        _print("=" * 60)
        _print("FULL REGISTRATION SUMMARY")
        _print(f"  Coarse RMSE: {coarse['error']:.4f} mm")
        _print(f"  Refined RMSE: {np.sqrt(fine['error']):.4f} mm  "
               f"({fine['iterations']} iters, converged={fine['converged']})")
        _print("=" * 60)

        return {
            "transform": fine["transform"],
            "coarse_transform": coarse["transform"],
            "coarse_error": coarse["error"],
            "fine_error": fine["error"],
            "iterations": fine["iterations"],
            "converged": fine["converged"],
            "stage": "full",
        }

    def get_debug_data(self) -> dict:
        """Return intermediate data for debug visualization."""
        return self._debug_data.copy()

    def _compute_rmse(
        self, src: np.ndarray, tgt: np.ndarray, T: np.ndarray,
    ) -> float:
        """Root mean squared error between transformed source and nearest target."""
        if len(src) == 0 or len(tgt) == 0:
            return float("inf")
        transformed = affine_solver.apply(T, src)
        from scipy.spatial import cKDTree
        tree = cKDTree(tgt)
        dists, _ = tree.query(transformed)
        return float(np.sqrt(np.mean(dists ** 2)))

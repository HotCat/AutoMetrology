"""
RegistrationPipeline — orchestrates the CAD-to-image alignment process.

Pipeline stages:
  1. Coarse: centroid/bbox matching → initial affine estimate
  2. Fine: ICP refinement using sampled CAD points and image edges

The pipeline is compute-only (no Qt deps). Results are returned as dicts
with numpy arrays. UI integration happens via signals in MainWindow.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Optional

from ..models.repository import FeatureRepository
from ..models.registration import RegistrationManager
from .sampler import CADFeatureSampler
from .image_extractor import ImageFeatureExtractor
from .icp_engine import ICPRegistrationEngine
from . import affine_solver

logger = logging.getLogger(__name__)


class RegistrationPipeline:
    """Orchestrates coarse-to-fine CAD-to-image registration."""

    def __init__(
        self,
        repo: FeatureRepository,
        reg_manager: RegistrationManager,
    ) -> None:
        self._repo = repo
        self._reg_manager = reg_manager
        self._sampler = CADFeatureSampler(default_density=1.0)
        self._extractor = ImageFeatureExtractor()
        self._icp = ICPRegistrationEngine(
            max_iterations=100, tolerance=1e-6, outlier_distance=10.0,
        )
        self._debug_data: dict = {}

    def run_coarse(
        self,
        image_path: str,
        group_id: str,
        pixel_size_mm: float,
    ) -> dict:
        """
        Coarse registration: centroid/bbox alignment.

        Args:
            image_path: path to telecentric image (PNG/BMP/TIF)
            group_id: registration group to use for alignment
            pixel_size_mm: mm per pixel (from camera calibration)

        Returns dict with 'transform', 'stage', 'error'.
        """
        group = self._reg_manager.get_group(group_id)
        if not group or not group.feature_ids:
            return {"transform": affine_solver.identity(), "stage": "coarse", "error": float("inf")}

        # Sample CAD features
        cad_points = self._sampler.sample_group(group, self._repo)
        if len(cad_points) < 3:
            return {"transform": affine_solver.identity(), "stage": "coarse", "error": float("inf")}

        # Load image and extract edges
        image = self._extractor.load_image(image_path)
        image_edges = self._extractor.extract_edges(image)
        if len(image_edges) < 3:
            return {"transform": affine_solver.identity(), "stage": "coarse", "error": float("inf")}

        # Convert image edges from pixel to world coords using pixel_size_mm
        # Note: image Y is inverted relative to CAD Y
        image_world = image_edges * pixel_size_mm
        image_world[:, 1] = -image_world[:, 1]  # flip Y

        # Compute centroids
        cad_centroid = self._sampler.compute_centroid(cad_points)
        img_centroid = self._sampler.compute_centroid(image_world)

        # Translation to align centroids
        T_translate = affine_solver.solve_from_centroids(cad_centroid, img_centroid)

        # Scale from bounding box sizes
        cad_bbox = self._sampler.compute_bbox(cad_points)
        img_bbox = self._sampler.compute_bbox(image_world)
        T_scale = affine_solver.solve_from_bbox(cad_bbox, img_bbox)

        # Compose: translate then scale
        T_coarse = affine_solver.compose([T_translate, T_scale])

        error = self._compute_rmse(cad_points, image_world, T_coarse)

        self._debug_data["coarse"] = {
            "cad_points": cad_points,
            "image_edges": image_edges,
            "image_world": image_world,
            "transform": T_coarse,
            "cad_centroid": cad_centroid,
            "img_centroid": img_centroid,
        }

        logger.info(
            f"Coarse registration: error={error:.4f}mm, "
            f"cad_pts={len(cad_points)}, img_pts={len(image_edges)}"
        )

        return {"transform": T_coarse, "stage": "coarse", "error": error}

    def run_fine(
        self,
        coarse_transform: np.ndarray,
        group_id: str,
        pixel_size_mm: Optional[float] = None,
    ) -> dict:
        """
        Fine registration: ICP refinement.

        Uses coarse transform as initial estimate for ICP.
        """
        group = self._reg_manager.get_group(group_id)
        if not group or not group.feature_ids:
            return {
                "transform": coarse_transform, "stage": "fine",
                "iterations": 0, "error": float("inf"), "converged": False,
            }

        cad_points = self._sampler.sample_group(group, self._repo)
        if len(cad_points) < 3:
            return {
                "transform": coarse_transform, "stage": "fine",
                "iterations": 0, "error": float("inf"), "converged": False,
            }

        # Use cached image edges from coarse stage
        coarse_data = self._debug_data.get("coarse", {})
        image_world = coarse_data.get("image_world")
        if image_world is None or len(image_world) < 3:
            return {
                "transform": coarse_transform, "stage": "fine",
                "iterations": 0, "error": float("inf"), "converged": False,
            }

        # Run ICP
        result = self._icp.align(cad_points, image_world, coarse_transform)

        self._debug_data["fine"] = {
            "transform": result["transform"],
            "iterations": result["iterations"],
            "error": result["final_error"],
            "correspondences": result["correspondences"],
        }

        logger.info(
            f"Fine registration: iterations={result['iterations']}, "
            f"error={result['final_error']:.4f}mm, "
            f"converged={result['converged']}"
        )

        return {
            "transform": result["transform"],
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
        """Run complete coarse → fine registration pipeline."""
        coarse = self.run_coarse(image_path, group_id, pixel_size_mm)
        if coarse["error"] == float("inf"):
            return coarse

        fine = self.run_fine(coarse["transform"], group_id, pixel_size_mm)

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
        self, src: np.ndarray, tgt: np.ndarray, T: np.ndarray
    ) -> float:
        """Root mean squared error between transformed source and nearest target."""
        if len(src) == 0 or len(tgt) == 0:
            return float("inf")
        transformed = affine_solver.apply(T, src)
        from scipy.spatial import cKDTree
        tree = cKDTree(tgt)
        dists, _ = tree.query(transformed)
        return float(np.sqrt(np.mean(dists ** 2)))

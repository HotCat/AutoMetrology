"""
RegistrationPipeline — orchestrator that delegates to a pluggable strategy.

The pipeline holds the shared state (repo, reg_manager, debug_data) and
delegates actual registration work to a RegistrationStrategy object.
The strategy is selectable at runtime via set_strategy().

Public API (unchanged from monolithic version):
  run_coarse(image_path, group_id, pixel_size_mm) -> dict
  run_fine(coarse_transform, group_id, pixel_size_mm) -> dict
  run_full(image_path, group_id, pixel_size_mm) -> dict
  get_debug_data() -> dict
  set_strategy(strategy) / get_strategy()
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from ..models.repository import FeatureRepository
from ..models.registration import RegistrationManager
from .strategy import (
    RegistrationStrategy,
    FullSilhouetteStrategy,
    ConvexHullStrategy,
    RegistrationContext,
    STRATEGY_REGISTRY,
)
from . import affine_solver


class RegistrationPipeline:
    """Strategy-based registration pipeline.

    Delegates to a RegistrationStrategy for the actual computation.
    The default strategy is FullSilhouetteStrategy (the original method).
    """

    def __init__(
        self,
        repo: FeatureRepository,
        reg_manager: RegistrationManager,
    ) -> None:
        self._repo = repo
        self._reg_manager = reg_manager
        self._strategy: RegistrationStrategy = FullSilhouetteStrategy()
        self._debug_data: dict = {}

    def set_strategy(self, strategy: RegistrationStrategy) -> None:
        """Switch the active registration strategy."""
        self._strategy = strategy

    def set_strategy_by_key(self, key: str) -> None:
        """Switch strategy by registry key (e.g. 'full_silhouette', 'convex_hull')."""
        cls = STRATEGY_REGISTRY.get(key)
        if cls is not None:
            self._strategy = cls()

    def get_strategy(self) -> RegistrationStrategy:
        """Return the current strategy."""
        return self._strategy

    def run_coarse(
        self,
        image_path: str,
        group_id: str,
        pixel_size_mm: float,
        anchor_handles: list[str] | None = None,
    ) -> dict:
        """Coarse registration. Returns dict with 'transform', 'stage', 'error'."""
        ctx = RegistrationContext(
            repo=self._repo,
            reg_manager=self._reg_manager,
            group_id=group_id,
            image_path=image_path,
            pixel_size_mm=pixel_size_mm,
            debug_data=self._debug_data,
            anchor_handles=anchor_handles or [],
        )
        result = self._strategy.run_coarse(ctx)
        return {
            "transform": result.transform,
            "stage": result.stage,
            "error": result.error,
        }

    def run_fine(
        self,
        coarse_transform: np.ndarray,
        group_id: str,
        pixel_size_mm: Optional[float] = None,
    ) -> dict:
        """Fine registration. Returns dict with 'transform', 'stage', 'iterations', 'error', 'converged'."""
        coarse_data = self._debug_data.get("coarse", {})
        ctx = RegistrationContext(
            repo=self._repo,
            reg_manager=self._reg_manager,
            group_id=group_id,
            image_path=coarse_data.get("image_path", ""),
            pixel_size_mm=coarse_data.get("pixel_size_mm", pixel_size_mm or 0.01),
            debug_data=self._debug_data,
        )
        result = self._strategy.run_fine(ctx, coarse_transform)
        return {
            "transform": result.transform,
            "stage": result.stage,
            "iterations": result.iterations,
            "error": result.error,
            "converged": result.converged,
        }

    def run_full(
        self,
        image_path: str,
        group_id: str,
        pixel_size_mm: float,
        anchor_handles: list[str] | None = None,
    ) -> dict:
        """Full coarse+fine registration. Returns dict matching original shape."""
        ctx = RegistrationContext(
            repo=self._repo,
            reg_manager=self._reg_manager,
            group_id=group_id,
            image_path=image_path,
            pixel_size_mm=pixel_size_mm,
            debug_data=self._debug_data,
            anchor_handles=anchor_handles or [],
        )
        result = self._strategy.run_full(ctx)
        return {
            "transform": result.transform,
            "coarse_transform": result.coarse_transform,
            "coarse_error": result.coarse_error,
            "fine_error": result.fine_error,
            "iterations": result.iterations,
            "converged": result.converged,
            "stage": result.stage,
        }

    def get_debug_data(self) -> dict:
        """Return intermediate data for debug visualization."""
        return self._debug_data.copy()

    def set_debug_data(self, key: str, data: dict) -> None:
        """Populate debug data cache (e.g. for manual alignment seeding)."""
        self._debug_data[key] = data

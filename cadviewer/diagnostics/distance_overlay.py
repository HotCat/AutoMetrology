"""
Distance Debug Overlay — draws measured lines, distance vectors, and distance
labels for every measurement, showing pixel distance and mm distance before
and after TPS.

This visualization is critical for identifying:
  - Which measurements have errors
  - Whether errors are consistent (scale) or variable (local distortion)
  - Whether TPS helps or hurts
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import numpy as np

from ..calibration.residual_map import ResidualDistortionMap
from ..registration import affine_solver

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@dataclass
class DistanceOverlayItem:
    """A single distance visualization item."""

    # Line A endpoints (pixels)
    line_a_p1: Tuple[float, float] = (0.0, 0.0)
    line_a_p2: Tuple[float, float] = (0.0, 0.0)
    line_a_color: Tuple[int, int, int] = (0, 255, 0)  # green

    # Line B endpoints (pixels)
    line_b_p1: Tuple[float, float] = (0.0, 0.0)
    line_b_p2: Tuple[float, float] = (0.0, 0.0)
    line_b_color: Tuple[int, int, int] = (0, 0, 255)  # red

    # Distance vector (from line A to line B)
    distance_vector_start: Tuple[float, float] = (0, 0)
    distance_vector_end: Tuple[float, float] = (0, 0)

    # Distance values
    pixel_distance_no_tps: float = 0.0
    pixel_distance_with_tps: float = 0.0
    mm_distance: float = 0.0
    nominal_mm: float = 0.0
    deviation_mm: float = 0.0

    # Labels
    label: str = ""
    label_position: Tuple[float, float] = (0, 0)


class DistanceOverlayRenderer:
    """Renders distance debug overlay on images."""

    def __init__(
        self,
        affine: np.ndarray,
        pixel_size_mm: float,
        residual_map: Optional[ResidualDistortionMap] = None,
    ) -> None:
        self._affine = affine
        self._pixel_size_mm = pixel_size_mm
        self._residual_map = residual_map

    def render_overlay(
        self,
        image: np.ndarray,
        items: List[DistanceOverlayItem],
    ) -> np.ndarray:
        """Render distance overlay on a copy of the image."""
        if not HAS_CV2:
            return image.copy()

        overlay = image.copy()
        if overlay.ndim == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

        for item in items:
            # Draw Line A (measured feature 1)
            p1 = (int(item.line_a_p1[0]), int(item.line_a_p1[1]))
            p2 = (int(item.line_a_p2[0]), int(item.line_a_p2[1]))
            cv2.line(overlay, p1, p2, item.line_a_color, 2)

            # Draw Line B (measured feature 2)
            p1 = (int(item.line_b_p1[0]), int(item.line_b_p1[1]))
            p2 = (int(item.line_b_p2[0]), int(item.line_b_p2[1]))
            cv2.line(overlay, p1, p2, item.line_b_color, 2)

            # Draw distance vector
            dv_start = (int(item.distance_vector_start[0]), int(item.distance_vector_start[1]))
            dv_end = (int(item.distance_vector_end[0]), int(item.distance_vector_end[1]))
            cv2.line(overlay, dv_start, dv_end, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(overlay, dv_start, 3, (255, 255, 0), -1)
            cv2.circle(overlay, dv_end, 3, (255, 255, 0), -1)

            # Draw label
            label_parts = []
            if item.label:
                label_parts.append(item.label)
            label_parts.append(f"px(no TPS): {item.pixel_distance_no_tps:.2f}")
            label_parts.append(f"px(TPS): {item.pixel_distance_with_tps:.2f}")
            label_parts.append(f"mm: {item.mm_distance:.4f}")
            label_parts.append(f"nominal: {item.nominal_mm:.4f}")
            label_parts.append(f"dev: {item.deviation_mm:+.4f}")

            dev_pct = (item.deviation_mm / item.nominal_mm * 100) if item.nominal_mm != 0 else 0
            label_parts.append(f"({dev_pct:+.2f}%)")

            # Choose color based on error magnitude
            if abs(dev_pct) > 5:
                text_color = (0, 0, 255)  # red
            elif abs(dev_pct) > 1:
                text_color = (0, 165, 255)  # orange
            else:
                text_color = (0, 255, 0)  # green

            lx, ly = int(item.label_position[0]), int(item.label_position[1])
            for i, part in enumerate(label_parts):
                cv2.putText(
                    overlay, part, (lx, ly + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA,
                )

        return overlay

    def compute_distance_items(
        self,
        mf1,
        mf2,
        nominal_mm: float,
        label: str = "",
    ) -> DistanceOverlayItem:
        """Compute distance overlay item from two measured features."""

        item = DistanceOverlayItem(label=label)

        # Extract pixel-space geometry
        g1_px = mf1.fitted_geometry
        g2_px = mf2.fitted_geometry

        # Compute pixel distances before TPS
        if "cx" in g1_px and "cx" in g2_px:
            # Circle distance
            p1 = (g1_px["cx"], g1_px["cy"])
            p2 = (g2_px["cx"], g2_px["cy"])
            item.pixel_distance_no_tps = math.sqrt(
                (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2
            )
            item.line_a_p1 = p1
            item.line_a_p2 = p1  # point, not line
            item.line_b_p1 = p2
            item.line_b_p2 = p2
            item.distance_vector_start = p1
            item.distance_vector_end = p2

        elif "x1" in g1_px and "x1" in g2_px:
            # Line distance
            item.line_a_p1 = (g1_px["x1"], g1_px["y1"])
            item.line_a_p2 = (g1_px["x2"], g1_px["y2"])
            item.line_b_p1 = (g2_px["x1"], g2_px["y1"])
            item.line_b_p2 = (g2_px["x2"], g2_px["y2"])

            # Compute perpendicular distance in pixels (before TPS)
            dx = g1_px["x2"] - g1_px["x1"]
            dy = g1_px["y2"] - g1_px["y1"]
            length = math.sqrt(dx * dx + dy * dy)
            if length > 0:
                nx, ny = -dy / length, dx / length
                mid_a = ((g1_px["x1"] + g1_px["x2"]) / 2,
                         (g1_px["y1"] + g1_px["y2"]) / 2)
                mid_b = ((g2_px["x1"] + g2_px["x2"]) / 2,
                         (g2_px["y1"] + g2_px["y2"]) / 2)

                d = abs((mid_b[0] - g1_px["x1"]) * nx + (mid_b[1] - g1_px["y1"]) * ny)
                item.pixel_distance_no_tps = d

                item.distance_vector_start = mid_a
                # Project mid_b onto the normal from mid_a
                proj_x = mid_a[0] + d * nx
                proj_y = mid_a[1] + d * ny
                item.distance_vector_end = (proj_x, proj_y)

        # Compute pixel distances after TPS
        if self._residual_map is not None and self._residual_map.is_built:
            if "cx" in g1_px and "cx" in g2_px:
                pts = np.array([[g1_px["cx"], g1_px["cy"]], [g2_px["cx"], g2_px["cy"]]])
                corrected = self._residual_map.correct(pts)
                item.pixel_distance_with_tps = math.sqrt(
                    (corrected[1, 0] - corrected[0, 0]) ** 2 +
                    (corrected[1, 1] - corrected[0, 1]) ** 2
                )
            elif "x1" in g1_px and "x1" in g2_px:
                # Correct all 4 endpoints
                pts = np.array([
                    [g1_px["x1"], g1_px["y1"]], [g1_px["x2"], g1_px["y2"]],
                    [g2_px["x1"], g2_px["y1"]], [g2_px["x2"], g2_px["y2"]],
                ])
                corrected = self._residual_map.correct(pts)

                dx = corrected[1, 0] - corrected[0, 0]
                dy = corrected[1, 1] - corrected[0, 1]
                length = math.sqrt(dx * dx + dy * dy)
                if length > 0:
                    nx, ny = -dy / length, dx / length
                    mid_b = ((corrected[2, 0] + corrected[3, 0]) / 2,
                             (corrected[2, 1] + corrected[3, 1]) / 2)
                    d = abs((mid_b[0] - corrected[0, 0]) * nx +
                            (mid_b[1] - corrected[0, 1]) * ny)
                    item.pixel_distance_with_tps = d
        else:
            item.pixel_distance_with_tps = item.pixel_distance_no_tps

        # MM distance from world coordinates
        w1 = mf1.fitted_geometry_world
        w2 = mf2.fitted_geometry_world

        if "cx" in w1 and "cx" in w2:
            item.mm_distance = math.sqrt(
                (w2["cx"] - w1["cx"]) ** 2 + (w2["cy"] - w1["cy"]) ** 2
            )
        elif "x1" in w1 and "x1" in w2:
            x1, y1 = w1["x1"], w1["y1"]
            x2, y2 = w1["x2"], w1["y2"]
            dx, dy = x2 - x1, y2 - y1
            length = math.sqrt(dx * dx + dy * dy)
            if length > 0:
                nx, ny = -dy / length, dx / length
                lx1, ly1 = w2["x1"], w2["y1"]
                lx2, ly2 = w2["x2"], w2["y2"]
                d1 = abs((lx1 - x1) * nx + (ly1 - y1) * ny)
                d2 = abs((lx2 - x1) * nx + (ly2 - y1) * ny)
                item.mm_distance = (d1 + d2) / 2

        item.nominal_mm = nominal_mm
        item.deviation_mm = item.mm_distance - nominal_mm

        # Label position: midpoint of distance vector
        dvx = (item.distance_vector_start[0] + item.distance_vector_end[0]) / 2
        dvy = (item.distance_vector_start[1] + item.distance_vector_end[1]) / 2
        item.label_position = (dvx + 10, dvy - 20)

        return item

    def render_measurement_debug(
        self,
        image: np.ndarray,
        measured_features: list,
        results: list,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """Render a full measurement debug overlay image."""

        items = []

        for result in results:
            if result.status != "ok" or result.instruction is None:
                continue

            fid1 = result.instruction.feature_id_1
            fid2 = result.instruction.feature_id_2

            # Find measured features
            mf1 = None
            mf2 = None
            for mf in measured_features:
                if mf.cad_feature_id == fid1:
                    mf1 = mf
                elif mf.cad_feature_id == fid2:
                    mf2 = mf

            if mf1 is None or mf2 is None:
                continue

            item = self.compute_distance_items(
                mf1, mf2,
                nominal_mm=result.nominal,
                label=f"lines({fid1},{fid2})" if "x1" in mf1.fitted_geometry else f"circles({fid1},{fid2})",
            )
            items.append(item)

        overlay = self.render_overlay(image, items)

        if save_path and HAS_CV2:
            cv2.imwrite(save_path, overlay)
            print(f"[OVERLAY] Saved to {save_path}")

        return overlay


def create_distance_overlay(
    image: np.ndarray,
    affine: np.ndarray,
    pixel_size_mm: float,
    measured_features: list,
    results: list,
    residual_map: Optional[ResidualDistortionMap] = None,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Create and optionally save a distance debug overlay image."""

    renderer = DistanceOverlayRenderer(affine, pixel_size_mm, residual_map)
    return renderer.render_measurement_debug(
        image, measured_features, results, save_path,
    )
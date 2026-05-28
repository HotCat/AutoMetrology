"""
Overlay renderers for registration groups and debug visualization.

RegistrationGroupOverlay — draws group boundaries, fills, and anchors.
DebugOverlay — draws CAD silhouette, image silhouette, minAreaRect,
               contour alignment, and local ROI boxes.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPainterPath, QPolygonF,
)

from ..models.feature import CADFeature, FeatureType
from ..models.repository import FeatureRepository
from ..models.registration import RegistrationGroup, RegistrationManager


class RegistrationGroupOverlay:
    """Renders visual overlays for registration groups on the canvas."""

    def __init__(self) -> None:
        self._label_font = QFont("Arial", 10, QFont.Bold)
        self._dash_pen = QPen()
        self._dash_pen.setStyle(Qt.DashLine)
        self._dash_pen.setWidth(1.5)

    def draw_group_overlays(
        self,
        painter: QPainter,
        groups: List[RegistrationGroup],
        repo: FeatureRepository,
        world_to_screen: Callable[[float, float], Tuple[float, float]],
        scale: float,
        feature_map: Dict[str, CADFeature],
    ) -> None:
        for group in groups:
            if not group.feature_ids:
                continue
            self._draw_single_group(painter, group, repo, world_to_screen, scale)

    def _draw_single_group(
        self,
        painter: QPainter,
        group: RegistrationGroup,
        repo: FeatureRepository,
        world_to_screen: Callable[[float, float], Tuple[float, float]],
        scale: float,
    ) -> None:
        bbox = group.bbox(repo)
        if not bbox:
            return

        min_x, min_y, max_x, max_y = bbox
        sx1, sy1 = world_to_screen(min_x, max_y)
        sx2, sy2 = world_to_screen(max_x, min_y)

        rect = QRectF(
            min(sx1, sx2), min(sy1, sy2),
            abs(sx2 - sx1), abs(sy2 - sy1),
        )

        color = group.color

        # Semi-transparent fill
        fill_color = QColor(color)
        fill_color.setAlpha(30)
        painter.setBrush(QBrush(fill_color))
        painter.setPen(Qt.NoPen)
        painter.drawRect(rect)

        # Dashed boundary
        dash_color = QColor(color)
        dash_color.setAlpha(160)
        self._dash_pen.setColor(dash_color)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(self._dash_pen)
        painter.drawRect(rect)

        # Centroid marker (diamond)
        centroid = group.centroid(repo)
        if centroid:
            cx, cy = world_to_screen(centroid[0], centroid[1])
            diamond_size = 6
            diamond = QPolygonF([
                QPointF(cx, cy - diamond_size),
                QPointF(cx + diamond_size, cy),
                QPointF(cx, cy + diamond_size),
                QPointF(cx - diamond_size, cy),
            ])
            anchor_color = QColor(color)
            anchor_color.setAlpha(200)
            painter.setBrush(QBrush(anchor_color))
            painter.setPen(QPen(anchor_color.darker(120), 1))
            painter.drawPolygon(diamond)

            # Group label
            painter.setFont(self._label_font)
            painter.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 220)))
            painter.drawText(QPointF(cx + diamond_size + 4, cy - 2), group.name)
            painter.setPen(QPen(QColor(180, 180, 180, 160)))
            small_font = QFont("Arial", 8)
            painter.setFont(small_font)
            painter.drawText(
                QPointF(cx + diamond_size + 4, cy + 12),
                f"{group.feature_count} features",
            )


class DebugOverlay:
    """Renders debug visualization for silhouette-based registration."""

    def __init__(self) -> None:
        pass

    def draw_debug(
        self,
        painter: QPainter,
        debug_data: dict,
        world_to_screen: Callable[[float, float], Tuple[float, float]],
        scale: float,
    ) -> None:
        """Draw all debug layers for the new silhouette-based pipeline."""
        from ..registration import affine_solver

        coarse = debug_data.get("coarse", {})
        fine = debug_data.get("fine", {})

        cad_points = coarse.get("cad_points")
        cad_contour = coarse.get("cad_contour")
        img_contour_world = coarse.get("img_contour_world")
        T_coarse = coarse.get("transform")
        rect_info = coarse.get("rect_info", {})
        pixel_size_mm = coarse.get("pixel_size_mm", 0.01)

        # 1. Draw CAD silhouette points (green)
        if cad_points is not None and len(cad_points) > 0:
            pen = QPen(QColor(0, 220, 80, 160), 2)
            painter.setPen(pen)
            pts = cad_points
            if len(pts) > 500:
                idx = np.linspace(0, len(pts) - 1, 500, dtype=int)
                pts = pts[idx]
            for pt in pts:
                sx, sy = world_to_screen(pt[0], pt[1])
                painter.drawPoint(QPointF(sx, sy))

        # 2. Draw CAD silhouette contour (green, thicker)
        if cad_contour is not None and len(cad_contour) >= 3:
            pen = QPen(QColor(0, 255, 100, 200), 2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            path = QPainterPath()
            sx0, sy0 = world_to_screen(cad_contour[0, 0], cad_contour[0, 1])
            path.moveTo(sx0, sy0)
            for pt in cad_contour[1:]:
                sx, sy = world_to_screen(pt[0], pt[1])
                path.lineTo(sx, sy)
            path.closeSubpath()
            painter.drawPath(path)

        # 3. Draw image silhouette transformed to CAD world (red)
        if img_contour_world is not None and T_coarse is not None and len(img_contour_world) > 0:
            T_inv = np.linalg.inv(T_coarse)
            cad_from_img = affine_solver.apply(T_inv, img_contour_world)

            # Draw as contour line
            if len(cad_from_img) >= 3:
                pen = QPen(QColor(255, 80, 80, 180), 2)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                path = QPainterPath()
                # Subsample if too many
                pts = cad_from_img
                if len(pts) > 300:
                    idx = np.linspace(0, len(pts) - 1, 300, dtype=int)
                    pts = pts[idx]
                sx0, sy0 = world_to_screen(pts[0, 0], pts[0, 1])
                path.moveTo(sx0, sy0)
                for pt in pts[1:]:
                    sx, sy = world_to_screen(pt[0], pt[1])
                    path.lineTo(sx, sy)
                path.closeSubpath()
                painter.drawPath(path)

        # 4. Draw minAreaRect for CAD (yellow dashed)
        if rect_info:
            cad_center = rect_info.get("cad_center")
            cad_size = rect_info.get("cad_size")
            cad_angle = rect_info.get("cad_angle", 0)
            cad_offset = rect_info.get("cad_angle_offset", 0)
            if cad_center is not None and cad_size is not None:
                self._draw_rotated_rect(
                    painter, cad_center, cad_size,
                    cad_angle + cad_offset,
                    QColor(255, 255, 0, 180), world_to_screen, scale,
                    label="CAD minAreaRect",
                )

            # 5. Draw minAreaRect for image (cyan dashed)
            img_center = rect_info.get("img_center")
            img_size = rect_info.get("img_size")
            img_angle = rect_info.get("img_angle", 0)
            img_offset = rect_info.get("img_angle_offset", 0)
            if img_center is not None and img_size is not None and T_coarse is not None:
                # Transform image rect center to CAD for display
                T_inv = np.linalg.inv(T_coarse)
                cad_center_from_img = affine_solver.apply(
                    T_inv, np.array([img_center]),
                )[0]
                # Scale the size by inverse transform
                img_size_cad = np.array(img_size) / affine_solver.extract_scale(T_coarse)
                self._draw_rotated_rect(
                    painter, cad_center_from_img, img_size_cad,
                    img_angle + img_offset,
                    QColor(0, 255, 255, 180), world_to_screen, scale,
                    label="Image minAreaRect",
                )

        # 6. Draw refinement contour alignment
        fine_cad = fine.get("cad_contour")
        fine_T = fine.get("transform", T_coarse)
        if fine_cad is not None and fine_T is not None and len(fine_cad) >= 3:
            transformed = affine_solver.apply(fine_T, fine_cad)
            T_inv = np.linalg.inv(fine_T)
            pen = QPen(QColor(0, 200, 255, 150), 1.5, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            path = QPainterPath()
            pts = transformed
            if len(pts) > 300:
                idx = np.linspace(0, len(pts) - 1, 300, dtype=int)
                pts = pts[idx]
            sx0, sy0 = world_to_screen(pts[0, 0], pts[0, 1])
            path.moveTo(sx0, sy0)
            for pt in pts[1:]:
                sx, sy = world_to_screen(pt[0], pt[1])
                path.lineTo(sx, sy)
            path.closeSubpath()
            painter.drawPath(path)

        # 7. Legend
        painter.setPen(QPen(QColor(200, 200, 200, 200)))
        painter.setFont(QFont("Arial", 9))
        lx, ly = 10, 10
        legend = [
            ("CAD silhouette points", QColor(0, 220, 80)),
            ("CAD silhouette contour", QColor(0, 255, 100)),
            ("Image silhouette (→CAD)", QColor(255, 80, 80)),
            ("CAD minAreaRect", QColor(255, 255, 0)),
            ("Image minAreaRect (→CAD)", QColor(0, 255, 255)),
            ("Refined contour alignment", QColor(0, 200, 255)),
        ]
        for text, color in legend:
            painter.setPen(QPen(color, 2))
            painter.drawLine(QPointF(lx, ly + 5), QPointF(lx + 20, ly + 5))
            painter.setPen(QPen(QColor(200, 200, 200, 200)))
            painter.drawText(QPointF(lx + 25, ly + 10), text)
            ly += 16

    def _draw_rotated_rect(
        self,
        painter: QPainter,
        center: np.ndarray,
        size: np.ndarray,
        angle_deg: float,
        color: QColor,
        world_to_screen: Callable,
        scale: float,
        label: str = "",
    ) -> None:
        """Draw a rotated rectangle in world coordinates."""
        w, h = float(size[0]), float(size[1])
        theta = math.radians(angle_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        # Four corners relative to center
        corners_local = [
            (-w / 2, -h / 2), (w / 2, -h / 2),
            (w / 2, h / 2), (-w / 2, h / 2),
        ]
        cx, cy = float(center[0]), float(center[1])
        corners_world = []
        for lx_, ly_ in corners_local:
            wx = cx + cos_t * lx_ - sin_t * ly_
            wy = cy + sin_t * lx_ + cos_t * ly_
            corners_world.append((wx, wy))

        pen = QPen(color, 1.5, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        polygon = QPolygonF()
        for wx, wy in corners_world:
            sx, sy = world_to_screen(wx, wy)
            polygon.append(QPointF(sx, sy))

        painter.drawPolygon(polygon)

        # Center marker
        scx, scy = world_to_screen(cx, cy)
        painter.setPen(QPen(color, 2))
        painter.drawLine(QPointF(scx - 4, scy), QPointF(scx + 4, scy))
        painter.drawLine(QPointF(scx, scy - 4), QPointF(scx, scy + 4))

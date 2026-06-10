"""
Overlay renderers for debug visualization.

DebugOverlay — draws CAD silhouette, image silhouette, minAreaRect,
               contour alignment.
MeasurementDebugOverlay — draws measured feature ROIs, edge points,
                          fitted circles/lines, confidence badges.
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

        # 4. Draw selected-line fitting correspondences for Teach+ICP.
        # Fitted points are image-world coordinates, so transform them back
        # through the final registration matrix before drawing in CAD space.
        line_fit = fine.get("line_fit", {})
        fine_T_for_lines = fine.get("transform", T_coarse)
        if line_fit.get("success") and fine_T_for_lines is not None:
            try:
                T_inv_lines = np.linalg.inv(fine_T_for_lines)
                fitted = line_fit.get("fitted_edge_points")
                predicted = line_fit.get("predicted_edge_points")
                cad_fit_pts = line_fit.get("cad_points")

                if fitted is not None and len(fitted) > 0:
                    fitted_cad = affine_solver.apply(T_inv_lines, fitted)
                    if len(fitted_cad) > 500:
                        idx = np.linspace(0, len(fitted_cad) - 1, 500, dtype=int)
                        fitted_cad = fitted_cad[idx]
                    pen = QPen(QColor(255, 60, 60, 210), 3)
                    painter.setPen(pen)
                    for pt in fitted_cad:
                        sx, sy = world_to_screen(pt[0], pt[1])
                        painter.drawPoint(QPointF(sx, sy))

                if predicted is not None and len(predicted) > 0:
                    predicted_cad = affine_solver.apply(T_inv_lines, predicted)
                    if len(predicted_cad) > 300:
                        idx = np.linspace(0, len(predicted_cad) - 1, 300, dtype=int)
                        predicted_cad = predicted_cad[idx]
                    pen = QPen(QColor(255, 190, 40, 170), 2)
                    painter.setPen(pen)
                    for pt in predicted_cad:
                        sx, sy = world_to_screen(pt[0], pt[1])
                        painter.drawPoint(QPointF(sx, sy))

                if cad_fit_pts is not None and len(cad_fit_pts) > 0:
                    pts = cad_fit_pts
                    if len(pts) > 300:
                        idx = np.linspace(0, len(pts) - 1, 300, dtype=int)
                        pts = pts[idx]
                    pen = QPen(QColor(60, 220, 255, 180), 2)
                    painter.setPen(pen)
                    for pt in pts:
                        sx, sy = world_to_screen(pt[0], pt[1])
                        painter.drawPoint(QPointF(sx, sy))
            except Exception:
                pass

        # 5. Draw minAreaRect for CAD (yellow dashed)
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

        # 7. Convex hull debug overlays (when hull strategy produced data)
        cad_hull = coarse.get("cad_hull")
        img_hull_world = coarse.get("img_hull_world")

        if cad_hull is not None and len(cad_hull) >= 3:
            # CAD convex hull (bright green, dash-dot)
            pen = QPen(QColor(100, 255, 100, 220), 2.5, Qt.DashDotLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            path = QPainterPath()
            sx0, sy0 = world_to_screen(cad_hull[0, 0], cad_hull[0, 1])
            path.moveTo(sx0, sy0)
            for pt in cad_hull[1:]:
                sx, sy = world_to_screen(pt[0], pt[1])
                path.lineTo(sx, sy)
            path.closeSubpath()
            painter.drawPath(path)

            # Hull vertex markers
            pen = QPen(QColor(100, 255, 100, 255), 4)
            painter.setPen(pen)
            for pt in cad_hull:
                sx, sy = world_to_screen(pt[0], pt[1])
                painter.drawPoint(QPointF(sx, sy))

        if (img_hull_world is not None and T_coarse is not None
                and len(img_hull_world) >= 3):
            T_inv = np.linalg.inv(T_coarse)
            hull_cad = affine_solver.apply(T_inv, img_hull_world)

            # Image hull in CAD coords (magenta, dash-dot)
            pen = QPen(QColor(255, 80, 255, 200), 2.5, Qt.DashDotLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            pts = hull_cad
            if len(pts) > 200:
                idx = np.linspace(0, len(pts) - 1, 200, dtype=int)
                pts = pts[idx]
            path = QPainterPath()
            sx0, sy0 = world_to_screen(pts[0, 0], pts[0, 1])
            path.moveTo(sx0, sy0)
            for pt in pts[1:]:
                sx, sy = world_to_screen(pt[0], pt[1])
                path.lineTo(sx, sy)
            path.closeSubpath()
            painter.drawPath(path)

        # 8. Legend
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
        # Add hull-specific legend entries when hull data is present
        if cad_hull is not None:
            legend.append(("CAD convex hull", QColor(100, 255, 100)))
        if img_hull_world is not None and T_coarse is not None:
            legend.append(("Image convex hull (→CAD)", QColor(255, 80, 255)))
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


class MeasurementDebugOverlay:
    """Renders measurement debug: ROI boxes, edge points, fitted geometry."""

    def __init__(self) -> None:
        pass

    def draw_measurement(
        self,
        painter: QPainter,
        meas_data: dict,
        world_to_screen: Callable[[float, float], Tuple[float, float]],
        scale: float,
        affine: np.ndarray,
    ) -> None:
        """Draw measurement overlays for all measured features.

        Args:
            meas_data: dict keyed by cad_feature_id, each value is a dict
                       with 'type', 'roi', 'edge_points', fitted geometry, etc.
            affine: 3x3 pixel→world affine (to transform pixel data to world)
            world_to_screen: world→screen coordinate transform
            scale: current zoom scale (pixels per mm)
        """
        from ..registration import affine_solver

        legend_y = 10

        for cad_id, data in meas_data.items():
            feat_type = data.get("type", "")
            confidence = data.get("confidence", 0.0)

            # Color based on confidence
            if confidence > 0.7:
                base_color = QColor(0, 200, 255)  # cyan (distinct from registration green)
            elif confidence > 0.4:
                base_color = QColor(255, 200, 0)  # yellow
            else:
                base_color = QColor(255, 80, 80)  # red

            # 1. Draw ROI box (dashed, transformed to world)
            roi = data.get("roi")
            if roi is not None:
                self._draw_roi_box(
                    painter, roi, affine, world_to_screen, scale,
                    QColor(base_color.red(), base_color.green(), base_color.blue(), 100),
                )

            # 2. Draw detected edge points (small cyan dots)
            edge_points = data.get("edge_points")
            if edge_points is not None and len(edge_points) > 0:
                pts_world = affine_solver.apply(affine, edge_points)
                pen = QPen(QColor(0, 200, 255, 220), 2)
                painter.setPen(pen)
                max_pts = 300
                draw_pts = pts_world
                if len(draw_pts) > max_pts:
                    idx = np.linspace(0, len(draw_pts) - 1, max_pts, dtype=int)
                    draw_pts = draw_pts[idx]
                for pt in draw_pts:
                    sx, sy = world_to_screen(pt[0], pt[1])
                    painter.drawPoint(QPointF(sx, sy))

            # 3. Draw fitted geometry
            if feat_type == "circle":
                self._draw_fitted_circle(
                    painter, data, affine, world_to_screen, scale, base_color,
                )
            elif feat_type == "line":
                self._draw_fitted_line(
                    painter, data, affine, world_to_screen, scale, base_color,
                )

        # Legend
        painter.setPen(QPen(QColor(200, 200, 200, 200)))
        painter.setFont(QFont("Arial", 9))
        lx = 10
        ly = legend_y
        legend = [
            ("Measured ROI", QColor(0, 180, 255, 100)),
            ("Edge points (fitted)", QColor(0, 200, 255)),
            ("Fitted circle/line", QColor(0, 180, 255)),
        ]
        for text, color in legend:
            painter.setPen(QPen(color, 2))
            painter.drawLine(QPointF(lx, ly + 5), QPointF(lx + 20, ly + 5))
            painter.setPen(QPen(QColor(200, 200, 200, 200)))
            painter.drawText(QPointF(lx + 25, ly + 10), text)
            ly += 16

    def _draw_roi_box(
        self,
        painter: QPainter,
        roi: tuple,
        affine: np.ndarray,
        world_to_screen: Callable,
        scale: float,
        color: QColor,
    ) -> None:
        """Draw ROI bounding box transformed to world coordinates."""
        from ..registration import affine_solver
        xmin, ymin, xmax, ymax = roi
        corners = np.array([
            [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax],
        ], dtype=np.float64)
        corners_world = affine_solver.apply(affine, corners)

        pen = QPen(color, 1, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        polygon = QPolygonF()
        for pt in corners_world:
            sx, sy = world_to_screen(pt[0], pt[1])
            polygon.append(QPointF(sx, sy))
        painter.drawPolygon(polygon)

    def _draw_fitted_circle(
        self,
        painter: QPainter,
        data: dict,
        affine: np.ndarray,
        world_to_screen: Callable,
        scale: float,
        color: QColor,
    ) -> None:
        """Draw fitted circle in world coordinates."""
        from ..registration import affine_solver
        center = data.get("fitted_center")
        radius = data.get("fitted_radius")
        if center is None or radius is None:
            return

        # Transform center to world
        center_world = affine_solver.apply(affine, center.reshape(1, 2))[0]
        # Transform radius point to get world radius
        edge_pt = np.array([[center[0] + radius, center[1]]])
        edge_world = affine_solver.apply(affine, edge_pt)[0]
        r_world = float(np.linalg.norm(edge_world - center_world))

        sx, sy = world_to_screen(center_world[0], center_world[1])
        r_screen = r_world * scale

        pen = QPen(color, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPointF(sx, sy), r_screen, r_screen)

        # Center crosshair
        pen2 = QPen(color, 1)
        painter.setPen(pen2)
        painter.drawLine(QPointF(sx - 4, sy), QPointF(sx + 4, sy))
        painter.drawLine(QPointF(sx, sy - 4), QPointF(sx, sy + 4))

    def _draw_fitted_line(
        self,
        painter: QPainter,
        data: dict,
        affine: np.ndarray,
        world_to_screen: Callable,
        scale: float,
        color: QColor,
    ) -> None:
        """Draw fitted line in world coordinates."""
        from ..registration import affine_solver
        p1 = data.get("fitted_p1")
        p2 = data.get("fitted_p2")
        if p1 is None or p2 is None:
            return

        pixel_pts = np.array([p1, p2])
        world_pts = affine_solver.apply(affine, pixel_pts)

        pen = QPen(color, 2)
        painter.setPen(pen)
        sx1, sy1 = world_to_screen(world_pts[0, 0], world_pts[0, 1])
        sx2, sy2 = world_to_screen(world_pts[1, 0], world_pts[1, 1])
        painter.drawLine(QPointF(sx1, sy1), QPointF(sx2, sy2))

        # Endpoint markers
        for sx, sy in [(sx1, sy1), (sx2, sy2)]:
            painter.drawEllipse(QPointF(sx, sy), 3, 3)

"""
Overlay renderers for registration groups and debug visualization.

RegistrationGroupOverlay — draws group boundaries, fills, and anchors.
DebugOverlay — draws edge points, correspondence links, ICP residuals.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

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
    """Renders debug visualization for registration and correspondence."""

    def __init__(self) -> None:
        self._edge_pen = QPen(QColor(0, 255, 0, 120), 1)
        self._link_pen_good = QPen(QColor(0, 255, 100, 150), 1, Qt.DashLine)
        self._link_pen_bad = QPen(QColor(255, 50, 50, 150), 1, Qt.DashLine)
        self._roi_pen = QPen(QColor(255, 255, 0, 100), 1, Qt.DotLine)

    def draw_edge_points(
        self,
        painter: QPainter,
        points,  # np.ndarray Nx2, world coords
        world_to_screen: Callable,
        color: QColor = None,
    ) -> None:
        if points is None or len(points) == 0:
            return
        pen = QPen(color or QColor(0, 255, 0, 100), 1)
        painter.setPen(pen)
        for pt in points:
            sx, sy = world_to_screen(pt[0], pt[1])
            painter.drawPoint(QPointF(sx, sy))

    def draw_correspondence_links(
        self,
        painter: QPainter,
        correspondences: list,
        world_to_screen: Callable,
        feature_map: Dict[str, CADFeature],
    ) -> None:
        for corr in correspondences:
            feat = feature_map.get(corr.cad_feature_id)
            if not feat:
                continue
            g = feat.geometry
            ft = feat.feature_type
            if ft == FeatureType.LINE:
                sx, sy = world_to_screen(
                    (g["x1"] + g["x2"]) / 2, (g["y1"] + g["y2"]) / 2
                )
            elif ft in (FeatureType.CIRCLE, FeatureType.ARC):
                sx, sy = world_to_screen(g["cx"], g["cy"])
            elif ft == FeatureType.POLYLINE:
                pts = g.get("points", [])
                if pts:
                    mid = pts[len(pts) // 2]
                    sx, sy = world_to_screen(mid[0], mid[1])
                else:
                    continue
            else:
                continue

            # Draw confidence indicator
            confidence = getattr(corr, "confidence", 0)
            if confidence >= 0.7:
                pen = self._link_pen_good
            else:
                pen = self._link_pen_bad
            painter.setPen(pen)
            painter.drawText(QPointF(sx + 5, sy - 5), f"{confidence:.0%}")

    def draw_roi_boxes(
        self,
        painter: QPainter,
        rois: list,
        world_to_screen: Callable,
    ) -> None:
        painter.setPen(self._roi_pen)
        painter.setBrush(Qt.NoBrush)
        for roi in rois:
            min_x, min_y, max_x, max_y = roi
            sx1, sy1 = world_to_screen(min_x, max_y)
            sx2, sy2 = world_to_screen(max_x, min_y)
            rect = QRectF(
                min(sx1, sx2), min(sy1, sy2),
                abs(sx2 - sx1), abs(sy2 - sy1),
            )
            painter.drawRect(rect)

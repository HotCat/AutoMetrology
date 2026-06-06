"""
CADViewerCanvas — high-performance 2D QPainter-based CAD viewer.

Primary rendering backend for the metrology inspection tool.
Renders DXF geometry (lines, arcs, circles, polylines, splines) using
QPainter with world-to-screen coordinate transformation.

Performance strategy:
  - Offscreen pixmap cache for static geometry
  - Frustum culling: skip offscreen features
  - Highlighted features rendered on top of cached base
  - Grid drawn at adaptive spacing

Coordinate System:
  - World coordinates = DXF units (mm)
  - Screen coordinates = widget pixels
  - Transform: screen = (world - offset) * scale + center
"""

from __future__ import annotations

import math
import numpy as np
from typing import Dict, List, Optional, Set, Tuple

from PySide6.QtCore import Qt, Signal, QPoint, QRectF, QPointF, QSize
from PySide6.QtGui import (
    QPainter, QPen, QColor, QBrush, QTransform, QPainterPath,
    QWheelEvent, QMouseEvent, QFont, QPolygonF, QPixmap,
)
from PySide6.QtWidgets import QWidget, QSizePolicy

from ..models.feature import CADFeature, FeatureType
from ..models.repository import FeatureRepository
from ..models.registration import RegistrationManager
from ..renderers.overlay_renderer import RegistrationGroupOverlay, DebugOverlay, MeasurementDebugOverlay
from ..renderers.image_layer import ImageLayerRenderer
from ..core.signals import bus


class CADViewerCanvas(QWidget):
    """2D CAD viewer using QPainter with pan/zoom/select."""

    feature_clicked = Signal(str)   # feature_id

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # View transform state
        self._scale = 1.0
        self._offset_x = 0.0      # world offset (center of view)
        self._offset_y = 0.0
        self._panning = False
        self._last_mouse: Optional[QPoint] = None

        # Data
        self._features: List[CADFeature] = []
        self._feature_map: Dict[str, CADFeature] = {}
        self._highlighted_ids: Set[str] = set()

        # Pixmap cache for base geometry
        self._cache_pixmap: Optional[QPixmap] = None
        self._cache_key: Optional[Tuple[float, float, float, int, int]] = None
        self._cache_dirty = True

        # Rendering settings
        self._bg_color = QColor(18, 18, 26)
        self._default_pen = QPen(QColor(200, 200, 200), 1.0)
        self._highlight_pen = QPen(QColor(0, 220, 255), 3.5)
        self._highlight_pen_outer = QPen(QColor(0, 140, 200, 100), 7.0)
        self._highlight_fill = QColor(0, 220, 255, 30)
        self._grid_color = QColor(40, 40, 52)

        # Bounding box for fit-all
        self._bbox_min = (0.0, 0.0)
        self._bbox_max = (1.0, 1.0)

        # World-space bounding boxes per feature for culling
        self._feature_bboxes: Dict[str, Tuple[float, float, float, float]] = {}

        # Registration group overlay
        self._reg_manager: Optional[RegistrationManager] = None
        self._group_overlay = RegistrationGroupOverlay()
        self._show_groups = True

        # Image layer
        self._image_layer = ImageLayerRenderer()

        # Debug overlay
        self._debug_mode = False
        self._debug_overlay = DebugOverlay()
        self._debug_data: dict = {}

        # Measurement debug overlay
        self._meas_debug_overlay = MeasurementDebugOverlay()
        self._meas_debug_data: dict = {}
        self._meas_debug_affine: Optional[np.ndarray] = None

        # Teach mode state
        self._teach_mode: bool = False
        self._teach_phase: str = ""  # "cad_p1", "cad_p2", "img_p1", "img_p2", "done"
        self._teach_cad_points: list = []   # [{label, world: [x,y]}]
        self._teach_img_points: list = []   # [{label, pixel: [x,y]}]

        # Connect signals
        bus.highlight_feature.connect(self._on_highlight_feature)
        bus.unhighlight_all.connect(self._on_unhighlight_all)
        bus.view_fit_all.connect(self.fit_all)
        bus.view_fit_feature.connect(self._on_fit_feature)
        bus.group_created.connect(self._on_groups_changed)
        bus.group_deleted.connect(self._on_groups_changed)
        bus.group_contents_changed.connect(self._on_groups_changed)
        bus.groups_cleared.connect(self._on_groups_changed)

    # ── coordinate transforms ──────────────────────────────────────

    def _world_to_screen(self, wx: float, wy: float) -> Tuple[float, float]:
        """Convert world (DXF mm) to screen (pixel) coordinates."""
        cx = self.width() / 2.0
        cy = self.height() / 2.0
        sx = (wx - self._offset_x) * self._scale + cx
        sy = -(wy - self._offset_y) * self._scale + cy
        if not (math.isfinite(sx) and math.isfinite(sy)):
            return (0.0, 0.0)
        return sx, sy

    def _screen_to_world(self, sx: float, sy: float) -> Tuple[float, float]:
        """Convert screen (pixel) to world (DXF mm) coordinates."""
        cx = self.width() / 2.0
        cy = self.height() / 2.0
        wx = (sx - cx) / self._scale + self._offset_x
        wy = -((sy - cy) / self._scale) + self._offset_y
        return wx, wy

    # ── data loading ───────────────────────────────────────────────

    def load_repository(self, repo: FeatureRepository) -> None:
        """Load features from repository and fit view."""
        self._features = repo.all_features()
        self._feature_map = {f.feature_id: f for f in self._features}
        self._compute_bounding_boxes()
        self._cache_dirty = True
        self.fit_all()

    def _compute_bounding_boxes(self) -> None:
        """Compute per-feature and global bounding boxes."""
        gmin_x, gmin_y = float('inf'), float('inf')
        gmax_x, gmax_y = float('-inf'), float('-inf')

        for feat in self._features:
            pts = self._feature_points(feat)
            if pts:
                fmin_x = min(p[0] for p in pts)
                fmin_y = min(p[1] for p in pts)
                fmax_x = max(p[0] for p in pts)
                fmax_y = max(p[1] for p in pts)
                self._feature_bboxes[feat.feature_id] = (fmin_x, fmin_y, fmax_x, fmax_y)
                gmin_x = min(gmin_x, fmin_x)
                gmin_y = min(gmin_y, fmin_y)
                gmax_x = max(gmax_x, fmax_x)
                gmax_y = max(gmax_y, fmax_y)

        if gmin_x == float('inf'):
            gmin_x, gmin_y, gmax_x, gmax_y = 0, 0, 210, 297

        pad = max(gmax_x - gmin_x, gmax_y - gmin_y) * 0.03
        self._bbox_min = (gmin_x - pad, gmin_y - pad)
        self._bbox_max = (gmax_x + pad, gmax_y + pad)

    def _feature_points(self, feat: CADFeature) -> List[Tuple[float, float]]:
        """Extract bounding-relevant points from a feature."""
        g = feat.geometry
        if feat.feature_type == FeatureType.LINE:
            return [(g["x1"], g["y1"]), (g["x2"], g["y2"])]
        elif feat.feature_type == FeatureType.CIRCLE:
            cx, cy, r = g["cx"], g["cy"], g["radius"]
            return [(cx - r, cy - r), (cx + r, cy + r)]
        elif feat.feature_type == FeatureType.ARC:
            if "radius" in g:
                cx, cy, r = g["cx"], g["cy"], g["radius"]
                return [(cx - r, cy - r), (cx + r, cy + r)]
            elif "major_axis" in g:
                cx, cy = g["cx"], g["cy"]
                mx, my = g["major_axis"]
                ratio = g["ratio"]
                rx = math.sqrt(mx * mx + my * my)
                ry = rx * ratio
                return [(cx - rx, cy - ry), (cx + rx, cy + ry)]
            return [(g["cx"], g["cy"])]
        elif feat.feature_type == FeatureType.POLYLINE:
            return g.get("points", [])
        elif feat.feature_type == FeatureType.SPLINE:
            return g.get("control_points", []) or g.get("fit_points", [])
        elif feat.feature_type == FeatureType.TEXT:
            x, y = g["x"], g["y"]
            h = g.get("height", 2.5)
            return [(x - h, y - h), (x + h * 10, y + h * 2)]
        elif feat.feature_type == FeatureType.DIMENSION:
            return []
        elif feat.feature_type == FeatureType.POINT:
            return [(g["x"], g["y"])]
        elif feat.feature_type == FeatureType.HATCH:
            pts = []
            paths = g.get("paths", [])
            for path_edges in paths:
                for edge in path_edges:
                    start = edge.get("start")
                    end = edge.get("end")
                    if start:
                        pts.append(start)
                    if end:
                        pts.append(end)
                    if edge.get("type") == "Polyline":
                        pts.extend(edge.get("points", []))
            return pts
        elif feat.feature_type == FeatureType.LEADER:
            return g.get("points", [])
        return []

    # ── view controls ──────────────────────────────────────────────

    def fit_all(self) -> None:
        w = self.width()
        h = self.height()
        if w == 0 or h == 0:
            return
        dx = self._bbox_max[0] - self._bbox_min[0]
        dy = self._bbox_max[1] - self._bbox_min[1]
        if dx == 0 or dy == 0:
            return
        self._scale = min((w * 0.92) / dx, (h * 0.92) / dy)
        self._offset_x = (self._bbox_min[0] + self._bbox_max[0]) / 2.0
        self._offset_y = (self._bbox_min[1] + self._bbox_max[1]) / 2.0
        self._cache_dirty = True
        self.update()

    def _on_fit_feature(self, feature_id: str) -> None:
        """Zoom to fit a specific feature with generous padding."""
        bbox = self._feature_bboxes.get(feature_id)
        if not bbox:
            return
        fmin_x, fmin_y, fmax_x, fmax_y = bbox
        pad = max(fmax_x - fmin_x, fmax_y - fmin_y) * 5.0
        if pad < 5.0:
            pad = 30.0

        dx = (fmax_x - fmin_x) + pad * 2
        dy = (fmax_y - fmin_y) + pad * 2
        w, h = self.width(), self.height()
        self._scale = min((w * 0.8) / dx, (h * 0.8) / dy)
        self._offset_x = (fmin_x + fmax_x) / 2.0
        self._offset_y = (fmin_y + fmax_y) / 2.0
        self._cache_dirty = True
        self.update()

    # ── frustum culling ────────────────────────────────────────────

    def _is_visible(self, bbox: Tuple[float, float, float, float]) -> bool:
        """Check if a world-space bbox overlaps the visible viewport."""
        wl, wt = self._screen_to_world(0, 0)
        wr, wb = self._screen_to_world(self.width(), self.height())
        vmin_x, vmax_x = min(wl, wr), max(wl, wr)
        vmin_y, vmax_y = min(wt, wb), max(wt, wb)

        bmin_x, bmin_y, bmax_x, bmax_y = bbox
        return bmax_x >= vmin_x and bmin_x <= vmax_x and bmax_y >= vmin_y and bmin_y <= vmax_y

    # ── painting ───────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setRenderHint(QPainter.TextAntialiasing, True)

            # Background
            painter.fillRect(self.rect(), self._bg_color)

            # Image layer (under everything)
            if self._image_layer.has_image:
                self._image_layer.draw_image(
                    painter, self._world_to_screen,
                    self.width(), self.height(),
                )

            # Grid (lightweight, drawn every frame)
            self._draw_grid(painter)

            # Base geometry — from cache or rendered fresh
            self._render_base(painter)

            # Highlighted features drawn on top with glow effect
            if self._highlighted_ids:
                for fid in self._highlighted_ids:
                    feat = self._feature_map.get(fid)
                    if feat:
                        self._draw_feature(painter, feat, highlighted=True)

            # Registration group overlays
            if self._reg_manager and self._show_groups:
                self._group_overlay.draw_group_overlays(
                    painter, self._reg_manager.all_groups(),
                    self._reg_manager._repo, self._world_to_screen,
                    self._scale, self._feature_map,
                )

            # Origin marker
            self._draw_origin_marker(painter)

            # Debug overlay
            if self._debug_mode and self._debug_data:
                self._debug_overlay.draw_debug(
                    painter, self._debug_data,
                    self._world_to_screen, self._scale,
                )

            # Measurement debug overlay
            if self._meas_debug_data and self._meas_debug_affine is not None:
                self._meas_debug_overlay.draw_measurement(
                    painter, self._meas_debug_data,
                    self._world_to_screen, self._scale,
                    self._meas_debug_affine,
                )

            # Teach mode markers
            self._draw_teach_markers(painter)

            # Coordinate info
            self._draw_info_overlay(painter)
        finally:
            painter.end()

    def _render_base(self, painter: QPainter) -> None:
        """Render base geometry, using pixmap cache when possible."""
        # Check if cache is valid
        current_key = (
            round(self._offset_x, 2), round(self._offset_y, 2),
            round(self._scale, 4), self.width(), self.height()
        )

        if not self._cache_dirty and self._cache_key == current_key and self._cache_pixmap:
            painter.drawPixmap(0, 0, self._cache_pixmap)
            return

        # Render to offscreen pixmap
        sz = self.size()
        if sz.width() <= 0 or sz.height() <= 0:
            return
        pm = QPixmap(sz)
        pm.fill(Qt.transparent)
        pm_painter = QPainter(pm)
        try:
            pm_painter.setRenderHint(QPainter.Antialiasing, True)

            line_w = self._scaled_line_width()
            base_pen = QPen(QColor(200, 200, 200), line_w)

            for feat in self._features:
                fid = feat.feature_id
                # Skip highlighted (drawn separately on top)
                if fid in self._highlighted_ids:
                    continue
                # Frustum culling
                bbox = self._feature_bboxes.get(fid)
                if bbox and not self._is_visible(bbox):
                    continue

                color = self._feature_color(feat)
                base_pen.setColor(color)
                base_pen.setWidth(line_w)
                pm_painter.setPen(base_pen)
                # SOLID (filled polygon from dimension arrows)
                if (feat.feature_type == FeatureType.POLYLINE
                        and feat.geometry.get("is_solid", False)):
                    fill_color = QColor(color)
                    fill_color.setAlpha(200)
                    pm_painter.setBrush(QBrush(fill_color))
                else:
                    pm_painter.setBrush(Qt.NoBrush)
                try:
                    self._draw_feature_geometry(pm_painter, feat)
                except Exception:
                    pass
        finally:
            pm_painter.end()

        # Save cache
        self._cache_pixmap = pm
        self._cache_key = current_key
        self._cache_dirty = False

        painter.drawPixmap(0, 0, pm)

    def _draw_feature(self, painter: QPainter, feat: CADFeature, highlighted: bool = False) -> None:
        """Render a single feature with optional highlight effect."""
        if highlighted:
            # Outer glow
            painter.setPen(self._highlight_pen_outer)
            painter.setBrush(Qt.NoBrush)
            self._draw_feature_geometry(painter, feat)

            # Inner bright line
            painter.setPen(self._highlight_pen)
            painter.setBrush(QBrush(self._highlight_fill))
            self._draw_feature_geometry(painter, feat)
        else:
            color = self._feature_color(feat)
            painter.setPen(QPen(color, self._scaled_line_width()))
            painter.setBrush(Qt.NoBrush)
            self._draw_feature_geometry(painter, feat)

    def _draw_feature_geometry(self, painter: QPainter, feat: CADFeature) -> None:
        """Render just the geometry paths (no pen/brush setup)."""
        g = feat.geometry
        ftype = feat.feature_type

        if ftype == FeatureType.LINE:
            x1, y1 = self._world_to_screen(g["x1"], g["y1"])
            x2, y2 = self._world_to_screen(g["x2"], g["y2"])
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        elif ftype == FeatureType.CIRCLE:
            cx, cy = self._world_to_screen(g["cx"], g["cy"])
            r = g["radius"] * self._scale
            painter.drawEllipse(QPointF(cx, cy), r, r)

        elif ftype == FeatureType.ARC:
            self._draw_arc(painter, g)

        elif ftype == FeatureType.POLYLINE:
            pts = g.get("points", [])
            if len(pts) >= 2:
                screen_pts = [QPointF(*self._world_to_screen(p[0], p[1])) for p in pts]
                if g.get("closed", False) and len(screen_pts) > 2:
                    painter.drawPolygon(QPolygonF(screen_pts))
                else:
                    for i in range(len(screen_pts) - 1):
                        painter.drawLine(screen_pts[i], screen_pts[i + 1])

        elif ftype == FeatureType.SPLINE:
            self._draw_spline(painter, g)

        elif ftype == FeatureType.TEXT:
            self._draw_text(painter, g)

        elif ftype == FeatureType.POINT:
            px, py = self._world_to_screen(g["x"], g["y"])
            painter.drawPoint(QPointF(px, py))

        elif ftype == FeatureType.HATCH:
            self._draw_hatch(painter, g)

        elif ftype == FeatureType.LEADER:
            self._draw_leader(painter, g)

    def _draw_arc(self, painter: QPainter, g: dict) -> None:
        """Render an arc by tessellation into line segments.

        QPainterPath.arcTo() produces near-invisible output when the arc
        bounding rect is sub-pixel (screen radius < 1 px).  Tessellation
        avoids this: each line segment is rasterised independently and
        remains visible even at very low zoom levels.
        """
        if g.get("is_ellipse"):
            self._draw_elliptical_arc(painter, g)
            return

        cx, cy = self._world_to_screen(g["cx"], g["cy"])
        r = g["radius"] * self._scale
        start_deg = g["start_angle"]
        span_deg = g["end_angle"] - start_deg
        if span_deg < 0:
            span_deg += 360

        self._tessellate_arc(painter, cx, cy, r, start_deg, span_deg)

    def _tessellate_arc(
        self,
        painter: QPainter,
        cx: float,
        cy: float,
        r: float,
        start_deg: float,
        span_deg: float,
    ) -> None:
        """Tessellate and draw a circular arc as line segments.

        Args:
            painter: QPainter to draw with
            cx, cy: Screen coordinates of center
            r: Screen radius in pixels
            start_deg: Start angle in degrees
            span_deg: Sweep angle in degrees (positive = counterclockwise)
        """
        if r < 1e-6:
            return
        n_seg = max(16, int(abs(span_deg) / 5))
        theta = math.radians(start_deg)
        d_theta = math.radians(span_deg) / n_seg

        cos_t, sin_t = math.cos(theta), math.sin(theta)
        cos_d, sin_d = math.cos(d_theta), math.sin(d_theta)

        prev = QPointF(cx + r * cos_t, cy - r * sin_t)
        for _ in range(n_seg):
            cos_t, sin_t = (
                cos_t * cos_d - sin_t * sin_d,
                sin_t * cos_d + cos_t * sin_d,
            )
            cur = QPointF(cx + r * cos_t, cy - r * sin_t)
            painter.drawLine(prev, cur)
            prev = cur

    def _draw_elliptical_arc(self, painter: QPainter, g: dict) -> None:
        """Render an elliptical arc by parametric evaluation.

        DXF ELLIPSE parametric form:
            P(t) = center + cos(t) * major_axis + sin(t) * minor_axis
        where minor_axis = ratio * (-major_axis.y, major_axis.x)
        """
        cx, cy = g["cx"], g["cy"]
        mx, my = g["major_axis"]
        ratio = g["ratio"]
        start_t = g["start_param"]
        end_t = g["end_param"]

        span = end_t - start_t
        if span < 0:
            span += 2 * math.pi
        if span < 1e-10:
            span = 2 * math.pi

        num_segments = max(24, int(span / (2 * math.pi) * 128))

        points = []
        for i in range(num_segments + 1):
            t = start_t + span * i / num_segments
            cos_t = math.cos(t)
            sin_t = math.sin(t)
            wx = cx + cos_t * mx - sin_t * my * ratio
            wy = cy + cos_t * my + sin_t * mx * ratio
            sx, sy = self._world_to_screen(wx, wy)
            points.append(QPointF(sx, sy))

        if len(points) >= 2:
            path = QPainterPath()
            path.moveTo(points[0])
            for p in points[1:]:
                path.lineTo(p)
            painter.drawPath(path)

    def _draw_hatch(self, painter: QPainter, g: dict) -> None:
        """Render hatch boundary edges."""
        paths = g.get("paths", [])
        for path_edges in paths:
            for edge in path_edges:
                etype = edge.get("type", "")
                if etype == "LineEdge":
                    start = edge.get("start")
                    end = edge.get("end")
                    if start and end:
                        sx, sy = self._world_to_screen(start[0], start[1])
                        ex, ey = self._world_to_screen(end[0], end[1])
                        painter.drawLine(QPointF(sx, sy), QPointF(ex, ey))
                elif etype == "ArcEdge":
                    center = edge.get("center")
                    radius = edge.get("radius")
                    start_deg = edge.get("start_angle")
                    end_deg = edge.get("end_angle")
                    if center and radius and start_deg is not None and end_deg is not None:
                        cx, cy = self._world_to_screen(center[0], center[1])
                        r = radius * self._scale
                        span = end_deg - start_deg
                        if span < 0:
                            span += 360
                        if span < 1e-10:
                            span = 360
                        self._tessellate_arc(painter, cx, cy, r, start_deg, span)
                elif etype == "EllipseEdge":
                    emx = edge.get("major_axis")
                    eratio = edge.get("ratio")
                    estart = edge.get("start")
                    eend = edge.get("end")
                    if emx and eratio and estart and eend:
                        sx, sy = self._world_to_screen(estart[0], estart[1])
                        ex, ey = self._world_to_screen(eend[0], eend[1])
                        painter.drawLine(QPointF(sx, sy), QPointF(ex, ey))
                elif etype == "SplineEdge":
                    ctrl = edge.get("control_points", [])
                    fit = edge.get("fit_points", [])
                    eval_pts = fit if fit else ctrl
                    if len(eval_pts) >= 2:
                        pts = self._interpolate_spline(eval_pts, edge.get("degree", 3))
                        screen_pts = [QPointF(*self._world_to_screen(p[0], p[1])) for p in pts]
                        path = QPainterPath()
                        path.moveTo(screen_pts[0])
                        for sp in screen_pts[1:]:
                            path.lineTo(sp)
                        painter.drawPath(path)
                elif etype == "Polyline":
                    pts = edge.get("points", [])
                    if len(pts) >= 2:
                        screen_pts = [QPointF(*self._world_to_screen(p[0], p[1])) for p in pts]
                        for i in range(len(screen_pts) - 1):
                            painter.drawLine(screen_pts[i], screen_pts[i + 1])

    def _draw_leader(self, painter: QPainter, g: dict) -> None:
        """Render LEADER as polyline with arrowhead at first vertex."""
        pts = g.get("points", [])
        if len(pts) < 2:
            return

        screen_pts = [QPointF(*self._world_to_screen(p[0], p[1])) for p in pts]

        # Draw leader line segments
        path = QPainterPath()
        path.moveTo(screen_pts[0])
        for sp in screen_pts[1:]:
            path.lineTo(sp)
        painter.drawPath(path)

        # Draw arrowhead at first vertex (start of leader)
        if len(screen_pts) >= 2:
            p0 = screen_pts[0]
            p1 = screen_pts[1]
            dx = p1.x() - p0.x()
            dy = p1.y() - p0.y()
            length = math.sqrt(dx * dx + dy * dy)
            if length > 1e-6:
                # Arrow size proportional to view scale
                arrow_len = max(6, min(14, 10))
                ux, uy = dx / length, dy / length
                # Perpendicular
                nx, ny = -uy, ux
                tip = p0
                left = QPointF(tip.x() - ux * arrow_len + nx * arrow_len * 0.35,
                               tip.y() - uy * arrow_len + ny * arrow_len * 0.35)
                right = QPointF(tip.x() - ux * arrow_len - nx * arrow_len * 0.35,
                                tip.y() - uy * arrow_len - ny * arrow_len * 0.35)
                arrow = QPolygonF([tip, left, right])
                painter.drawPolygon(arrow)

    def _draw_spline(self, painter: QPainter, g: dict) -> None:
        """Render a spline using pre-evaluated points or Catmull-Rom fallback."""
        # Prefer pre-evaluated points from ezdxf's BSpline evaluator
        eval_pts = g.get("eval_points", [])
        if eval_pts and len(eval_pts) >= 2:
            screen_pts = [QPointF(*self._world_to_screen(p[0], p[1])) for p in eval_pts]
            path = QPainterPath()
            path.moveTo(screen_pts[0])
            for sp in screen_pts[1:]:
                path.lineTo(sp)
            painter.drawPath(path)
            return

        # Fallback to Catmull-Rom interpolation
        ctrl_pts = g.get("control_points", [])
        fit_pts = g.get("fit_points", [])
        eval_pts = fit_pts if fit_pts else ctrl_pts
        if len(eval_pts) < 2:
            return

        pts = self._interpolate_spline(eval_pts, g.get("degree", 3))
        screen_pts = [QPointF(*self._world_to_screen(p[0], p[1])) for p in pts]
        if len(screen_pts) >= 2:
            path = QPainterPath()
            path.moveTo(screen_pts[0])
            for sp in screen_pts[1:]:
                path.lineTo(sp)
            painter.drawPath(path)

    def _interpolate_spline(self, points: list, degree: int) -> list:
        """Catmull-Rom spline interpolation through control points."""
        if len(points) < 3 or degree < 3:
            return points

        result = []
        n = len(points)
        segments_per_span = 12
        for i in range(n - 1):
            p0 = points[max(i - 1, 0)]
            p1 = points[i]
            p2 = points[min(i + 1, n - 1)]
            p3 = points[min(i + 2, n - 1)]

            for s in range(segments_per_span):
                t = s / segments_per_span
                t2, t3 = t * t, t * t * t
                x = 0.5 * (
                    (2 * p1[0]) +
                    (-p0[0] + p2[0]) * t +
                    (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                    (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
                )
                y = 0.5 * (
                    (2 * p1[1]) +
                    (-p0[1] + p2[1]) * t +
                    (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                    (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
                )
                result.append((x, y))
        result.append(points[-1])
        return result

    def _draw_text(self, painter: QPainter, g: dict) -> None:
        """
        Render TEXT/MTEXT entities.

        Handles:
          - World-to-screen coordinate transform
          - Font sizing based on text height and current scale
          - Rotation (if specified)
          - DXF text formatting codes (%%c for diameter, \S for superscript, etc.)
        """
        raw_text = g.get("text", "")
        if not raw_text:
            return

        # Convert DXF text formatting codes to readable text
        clean_text = self._process_dxf_text_formatting(raw_text)

        # World position
        wx, wy = g.get("x", 0), g.get("y", 0)
        sx, sy = self._world_to_screen(wx, wy)

        # Font size: scale text height by current zoom
        text_height_world = g.get("height", 2.5)
        text_height_screen = text_height_world * self._scale

        # Clamp to reasonable pixel sizes
        font_size = max(6, min(text_height_screen, 48))

        font = QFont("Arial", int(font_size))
        font.setBold(True)
        painter.setFont(font)

        # Rotation (in degrees, DXF uses counterclockwise from X-axis)
        rotation = g.get("rotation", 0)

        # Save painter state and apply rotation
        if rotation != 0:
            painter.save()
            # DXF rotation: counterclockwise from X-axis
            # Qt rotation: clockwise, so negate and adjust for screen Y-flip
            painter.translate(QPointF(sx, sy))
            painter.rotate(-rotation)  # Apply rotation
            painter.drawText(0, 0, clean_text)
            painter.restore()
        else:
            painter.drawText(QPointF(sx, sy), clean_text)

    def _process_dxf_text_formatting(self, text: str) -> str:
        """
        Convert DXF text formatting codes to readable Unicode characters.

        Common codes:
          %%c    → Ø (diameter symbol)
          %%d    → ° (degree symbol)
          %%p    → ± (plus-minus tolerance)
          \S...^... → superscript/subscript tolerance notation
          \H0.5X  → height scaling (ignored for now)
          \A1;    → alignment (ignored)
          \P      → paragraph break → newline
          \L...\l → underline (stripped)
          \O...\o → overline (stripped)
          \K...\k → strikethrough (stripped)
        """
        import re

        result = text

        # Special character codes
        result = result.replace("%%c", "Ø")
        result = result.replace("%%C", "Ø")
        result = result.replace("%%d", "°")
        result = result.replace("%%D", "°")
        result = result.replace("%%p", "±")
        result = result.replace("%%P", "±")
        result = result.replace("%%u", "")
        result = result.replace("%%U", "")

        # Paragraph break
        result = result.replace("\\P", "\n")
        result = result.replace("\\p", "\n")

        # Superscript/subscript tolerance notation: \Svalue^tolerance;
        # Example: \S-0.01^ 0; → "-0.01/0" (tolerance above/below line)
        def replace_superscript(match):
            base = match.group(1).strip()
            sup = match.group(2).strip()
            if sup:
                return f"{base}/{sup}"
            return base

        # With trailing semicolon
        result = re.sub(r'\\S([^;^]*?)\^([^;]*?);', replace_superscript, result)
        # Without trailing semicolon
        result = re.sub(r'\\S([^;^]*?)\^([^;\\]*)', replace_superscript, result)

        # Remove remaining formatting codes
        # Height scaling \H...X; or \H...;
        result = re.sub(r'\\H[\d.]*X;?', '', result)
        result = re.sub(r'\\H[\d.]*;', '', result)
        # Alignment \A...;
        result = re.sub(r'\\A\d;', '', result)
        # Color \C\d;
        result = re.sub(r'\\C\d;', '', result)
        # Font name \F...;
        result = re.sub(r'\\F[^;]*;', '', result)
        # Underline, overline, strikethrough (toggle on/off)
        result = re.sub(r'\\[LlOoKk]', '', result)
        # Width/tracking \W...\T...
        result = re.sub(r'\\[WwTt][\d.]*;?', '', result)

        # Remove braces around plain text after formatting codes stripped
        result = re.sub(r'\{([^}]*)\}', r'\1', result)

        # Remove any remaining backslash sequences
        result = re.sub(r'\\[^\\;{}]*;?', '', result)

        return result.strip()

    def _draw_grid(self, painter: QPainter) -> None:
        """Draw adaptive grid lines."""
        pen = QPen(self._grid_color, 0.5, Qt.DotLine)
        painter.setPen(pen)

        base_spacing = 10.0
        if self._scale > 0:
            pixel_spacing = base_spacing * self._scale
            while pixel_spacing < 40:
                base_spacing *= 2
                pixel_spacing = base_spacing * self._scale
            while pixel_spacing > 160:
                base_spacing /= 2
                pixel_spacing = base_spacing * self._scale

        wl, wt = self._screen_to_world(0, 0)
        wr, wb = self._screen_to_world(self.width(), self.height())
        vmin_x, vmax_x = min(wl, wr), max(wl, wr)
        vmin_y, vmax_y = min(wt, wb), max(wt, wb)

        x = math.floor(vmin_x / base_spacing) * base_spacing
        while x <= vmax_x:
            sx, _ = self._world_to_screen(x, 0)
            painter.drawLine(QPointF(sx, 0), QPointF(sx, self.height()))
            x += base_spacing

        y = math.floor(vmin_y / base_spacing) * base_spacing
        while y <= vmax_y:
            _, sy = self._world_to_screen(0, y)
            painter.drawLine(QPointF(0, sy), QPointF(self.width(), sy))
            y += base_spacing

    def _draw_origin_marker(self, painter: QPainter) -> None:
        sx, sy = self._world_to_screen(0, 0)
        pen = QPen(QColor(100, 60, 60), 1)
        painter.setPen(pen)
        painter.drawLine(QPointF(sx - 10, sy), QPointF(sx + 10, sy))
        pen.setColor(QColor(60, 100, 60))
        painter.setPen(pen)
        painter.drawLine(QPointF(sx, sy - 10), QPointF(sx, sy + 10))

    def _draw_info_overlay(self, painter: QPainter) -> None:
        """Draw zoom level and cursor coordinates overlay."""
        font = QFont("Monospace", 9)
        painter.setFont(font)
        painter.setPen(QColor(120, 120, 140))
        zoom_pct = self._scale * 100
        painter.drawText(10, self.height() - 10, f"Zoom: {zoom_pct:.0f}%")

    def _feature_color(self, feat: CADFeature) -> QColor:
        dxf_colors = {
            0: QColor(180, 180, 180),
            1: QColor(255, 60, 60),
            2: QColor(255, 255, 60),
            3: QColor(60, 255, 60),
            4: QColor(60, 255, 255),
            5: QColor(60, 100, 255),
            6: QColor(255, 60, 255),
            7: QColor(180, 180, 180),
        }
        if feat.feature_type == FeatureType.HATCH:
            return QColor(45, 45, 60)
        if feat.feature_type == FeatureType.SPLINE:
            return QColor(180, 140, 255)
        if feat.feature_type == FeatureType.TEXT:
            return QColor(170, 220, 120)
        if feat.feature_type == FeatureType.LEADER:
            return QColor(255, 180, 80)
        return dxf_colors.get(feat.color % 256, QColor(180, 180, 180))

    def _scaled_line_width(self) -> float:
        return max(0.7, min(1.5, 1.0))

    # ── mouse interaction ──────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._last_mouse = event.pos()
        if self._teach_mode and event.button() == Qt.LeftButton:
            self._handle_teach_click(event.pos())
            return
        if event.button() == Qt.LeftButton:
            hit_id = self._hit_test(event.pos())
            if hit_id:
                self.feature_clicked.emit(hit_id)
                bus.highlight_feature.emit(hit_id)
                bus.property_update.emit({"feature_id": hit_id})
            else:
                bus.feature_deselected.emit()
                bus.unhighlight_all.emit()
        elif event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._panning = True
            self.setCursor(Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._panning = False
        if self._teach_mode:
            self.setCursor(Qt.CrossCursor)
            return
        self.setCursor(Qt.ArrowCursor)
        self._last_mouse = None

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._panning and self._last_mouse:
            dx = event.x() - self._last_mouse.x()
            dy = event.y() - self._last_mouse.y()
            self._offset_x -= dx / self._scale
            self._offset_y += dy / self._scale
            self._last_mouse = event.pos()
            self._cache_dirty = True
            self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        angle = event.angleDelta().y()
        factor = 1.15 if angle > 0 else 1.0 / 1.15

        mx, my = event.position().x(), event.position().y()
        wx, wy = self._screen_to_world(mx, my)

        self._scale *= factor

        cx = self.width() / 2.0
        cy = self.height() / 2.0
        self._offset_x = wx - (mx - cx) / self._scale
        self._offset_y = wy + (my - cy) / self._scale

        self._cache_dirty = True
        self.update()

    # ── teach mode point picking ───────────────────────────────────

    PHASE_LABELS = {
        "cad_p1": "P1", "cad_p2": "P2",
        "img_p1": "P1", "img_p2": "P2",
    }

    def _handle_teach_click(self, pos: QPoint) -> None:
        wx, wy = self._screen_to_world(pos.x(), pos.y())

        if self._teach_phase in ("cad_p1", "cad_p2"):
            label = self.PHASE_LABELS[self._teach_phase]
            self._teach_cad_points.append({"label": label, "world": [wx, wy]})
            bus.teach_point_added.emit({
                "phase": self._teach_phase,
                "world": [wx, wy],
            })
            if self._teach_phase == "cad_p1":
                self._teach_phase = "cad_p2"
            else:
                self._teach_phase = "img_p1"

        elif self._teach_phase in ("img_p1", "img_p2"):
            # Convert world click to image pixel coords
            from ..registration.affine_solver import apply as aff_apply
            inv_affine = np.linalg.inv(self._image_layer.affine)
            px_world = np.array([[wx, wy]], dtype=np.float64)
            px_img = aff_apply(inv_affine, px_world)
            pixel = [float(px_img[0, 0]), float(px_img[0, 1])]

            label = self.PHASE_LABELS[self._teach_phase]
            self._teach_img_points.append({"label": label, "pixel": pixel})
            bus.teach_point_added.emit({
                "phase": self._teach_phase,
                "pixel": pixel,
            })
            if self._teach_phase == "img_p1":
                self._teach_phase = "img_p2"
            else:
                self._teach_phase = "done"
                self._teach_mode = False
                self.setCursor(Qt.ArrowCursor)
                bus.teach_mode_completed.emit({
                    "cad_points": list(self._teach_cad_points),
                    "img_points": list(self._teach_img_points),
                })

        self.update()

    def _draw_teach_markers(self, painter: QPainter) -> None:
        if not self._teach_cad_points and not self._teach_img_points:
            return
        painter.save()
        marker_radius = max(6.0, 12.0 / self._scale)

        # Draw CAD point markers (green)
        cad_pen = QPen(QColor(0, 255, 100), 2)
        cad_font = painter.font()
        cad_font.setPixelSize(max(10, int(14 / self._scale)))
        painter.setFont(cad_font)
        for pt in self._teach_cad_points:
            wx, wy = pt["world"]
            painter.setPen(cad_pen)
            painter.drawEllipse(QPointF(wx, wy), marker_radius, marker_radius)
            painter.drawText(
                QPointF(wx + marker_radius * 1.5, wy - marker_radius * 0.5),
                pt["label"],
            )

        # Draw image point markers (magenta) — in world coords
        img_pen = QPen(QColor(255, 80, 255), 2)
        from ..registration.affine_solver import apply as aff_apply
        for pt in self._teach_img_points:
            px = np.array([pt["pixel"]], dtype=np.float64)
            world = aff_apply(self._image_layer.affine, px)
            wx, wy = float(world[0, 0]), float(world[0, 1])
            painter.setPen(img_pen)
            painter.drawEllipse(QPointF(wx, wy), marker_radius, marker_radius)
            painter.drawText(
                QPointF(wx + marker_radius * 1.5, wy - marker_radius * 0.5),
                pt["label"],
            )

        painter.restore()

    # ── hit testing ────────────────────────────────────────────────

    def _hit_test(self, pos: QPoint) -> Optional[str]:
        """Find the closest feature to a screen point."""
        wx, wy = self._screen_to_world(pos.x(), pos.y())
        hit_radius = 5.0 / self._scale

        best_dist = float('inf')
        best_id = None

        for feat in self._features:
            if feat.feature_type == FeatureType.HATCH:
                continue
            # Quick bbox pre-check
            bbox = self._feature_bboxes.get(feat.feature_id)
            if bbox:
                bmin_x, bmin_y, bmax_x, bmax_y = bbox
                # Expand bbox by hit radius for pre-filter
                if wx < bmin_x - hit_radius or wx > bmax_x + hit_radius:
                    continue
                if wy < bmin_y - hit_radius or wy > bmax_y + hit_radius:
                    continue

            dist = self._point_to_feature_distance(wx, wy, feat, hit_radius)
            if dist < hit_radius and dist < best_dist:
                best_dist = dist
                best_id = feat.feature_id

        return best_id

    def _point_to_feature_distance(self, px: float, py: float, feat: CADFeature, hit_radius: float = 5.0) -> float:
        """Minimum distance from point to feature geometry."""
        g = feat.geometry
        ftype = feat.feature_type

        if ftype == FeatureType.LINE:
            return self._pt_seg_dist(px, py, g["x1"], g["y1"], g["x2"], g["y2"])

        elif ftype == FeatureType.CIRCLE:
            d = math.sqrt((px - g["cx"]) ** 2 + (py - g["cy"]) ** 2)
            return abs(d - g["radius"])

        elif ftype == FeatureType.ARC:
            if "radius" not in g:
                # Elliptical arc — use center distance as approximation
                d = math.sqrt((px - g["cx"]) ** 2 + (py - g["cy"]) ** 2)
                return d
            dx, dy = px - g["cx"], py - g["cy"]
            d = math.sqrt(dx * dx + dy * dy)
            a = math.degrees(math.atan2(dy, dx)) % 360
            s = g["start_angle"] % 360
            e = g["end_angle"] % 360
            in_range = s <= a <= e if s <= e else (a >= s or a <= e)
            return abs(d - g["radius"]) if in_range else float('inf')

        elif ftype == FeatureType.POLYLINE:
            pts = g.get("points", [])
            md = float('inf')
            for i in range(len(pts) - 1):
                md = min(md, self._pt_seg_dist(px, py, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]))
            if g.get("closed") and len(pts) > 2:
                md = min(md, self._pt_seg_dist(px, py, pts[-1][0], pts[-1][1], pts[0][0], pts[0][1]))
            return md

        elif ftype == FeatureType.SPLINE:
            pts = g.get("control_points", []) or g.get("fit_points", [])
            if len(pts) < 2:
                return float('inf')
            md = float('inf')
            for i in range(len(pts) - 1):
                md = min(md, self._pt_seg_dist(px, py, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]))
            return md

        elif ftype == FeatureType.TEXT:
            tx, ty = g.get("x", 0), g.get("y", 0)
            th = g.get("height", 2.5)
            hit_margin = hit_radius * 2.0
            text_width_est = th * 5.0
            if abs(px - tx) < (text_width_est + hit_margin) and abs(py - ty) < (th + hit_margin):
                return 0.001
            return float('inf')

        elif ftype == FeatureType.LEADER:
            pts = g.get("points", [])
            md = float('inf')
            for i in range(len(pts) - 1):
                md = min(md, self._pt_seg_dist(px, py, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]))
            return md

        return float('inf')

    @staticmethod
    def _pt_seg_dist(px, py, x1, y1, x2, y2) -> float:
        dx, dy = x2 - x1, y2 - y1
        len_sq = dx * dx + dy * dy
        if len_sq < 1e-12:
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / len_sq))
        return math.sqrt((px - x1 - t * dx) ** 2 + (py - y1 - t * dy) ** 2)

    # ── signal handlers ────────────────────────────────────────────

    def _on_highlight_feature(self, feature_id: str) -> None:
        self._highlighted_ids = {feature_id}
        self._cache_dirty = True
        self.update()

    def _on_unhighlight_all(self) -> None:
        self._highlighted_ids.clear()
        self._cache_dirty = True
        self.update()

    # ── registration group integration ──────────────────────────────

    def set_registration_manager(self, manager: Optional[RegistrationManager]) -> None:
        self._reg_manager = manager
        self.update()

    def _on_groups_changed(self, *args) -> None:
        self._cache_dirty = True
        self.update()

    # ── image layer integration ────────────────────────────────────

    def get_image_layer(self) -> ImageLayerRenderer:
        return self._image_layer

    # ── teach mode ────────────────────────────────────────────────

    def start_teach_mode(self) -> None:
        self._teach_mode = True
        self._teach_phase = "cad_p1"
        self._teach_cad_points.clear()
        self._teach_img_points.clear()
        self.setCursor(Qt.CrossCursor)
        bus.teach_mode_started.emit()
        self.update()

    def cancel_teach_mode(self) -> None:
        self._teach_mode = False
        self._teach_phase = ""
        self._teach_cad_points.clear()
        self._teach_img_points.clear()
        self.setCursor(Qt.ArrowCursor)
        bus.teach_mode_cancelled.emit()
        self.update()

    def is_teach_mode(self) -> bool:
        return self._teach_mode

    @property
    def teach_phase(self) -> str:
        return self._teach_phase

    @property
    def teach_cad_points(self) -> list:
        return list(self._teach_cad_points)

    @property
    def teach_img_points(self) -> list:
        return list(self._teach_img_points)

    def set_debug_data(self, data: dict) -> None:
        """Store debug data from registration pipeline for overlay rendering."""
        self._debug_data = data
        if self._debug_mode:
            self.update()

    def set_debug_mode(self, enabled: bool) -> None:
        """Toggle debug overlay rendering."""
        self._debug_mode = enabled
        self.update()

    def set_measurement_debug(
        self, data: dict, affine: np.ndarray,
    ) -> None:
        """Store measurement debug data and the pixel→world affine."""
        self._meas_debug_data = data
        self._meas_debug_affine = affine
        self.update()

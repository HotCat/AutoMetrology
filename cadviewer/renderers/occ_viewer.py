"""
OCCViewerWidget — OpenCascade 2D/3D viewer embedded in a PySide6 QWidget.

This widget:
  - Creates an OCC V3d_View within a Qt window handle
  - Manages AIS_InteractiveContext for display/selection/hiding
  - Supports pan/zoom/fit-all mouse interactions
  - Renders features from geometry dict (conversion from parser layer)

DXF → OpenCascade geometry conversion:

  LINE → TopoDS_Edge via BRepBuilderAPI_MakeEdge(gp_Pnt(x1,y1,0), gp_Pnt(x2,y2,0))
  CIRCLE → TopoDS_Edge via BRepBuilderAPI_MakeEdge(Geom_Circle(gp_Circ(axis, radius)))
  ARC → TopoDS_Edge via GC_MakeArcOfCircle(center, start_angle, end_angle) → edge
  POLYLINE → TopoDS_Wire via BRepBuilderAPI_MakeWire from successive edges
  SPLINE → TopoDS_Edge via BRepBuilderAPI_MakeEdge(Geom_BSplineCurve(...))

Scene Graph Management:
  - AIS_InteractiveContext holds all AIS_Shape/AIS_InteractiveObject
  - Each displayed object has a AIS_KindOfInteractive type
  - Selection is managed by AIS_InteractiveContext::MoveTo/Select
  - Highlighting uses native AIS selection highlighting (color/linewidth)
"""

from __future__ import annotations

import math
import sys
import warnings
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal, QPoint, QTimer
from PySide6.QtGui import QMouseEvent, QWheelEvent, QResizeEvent
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSizePolicy

# OpenCascade imports - handle gracefully if not installed
try:
    from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Vec, gp_Ax1, gp_Ax2, gp_Circ, gp_XYZ
    from OCC.Core.Geom import Geom_Circle, Geom_Line, Geom_BSplineCurve
    from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
    from OCC.Core.GC import GC_MakeArcOfCircle, GC_MakeSegment
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
    )
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Edge, TopoDS_Wire, TopoDS_Compound
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_WIRE
    from OCC.Core.TColgp import TColgp_Array1OfPnt
    from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
    from OCC.Core.AIS import AIS_Shape, AIS_InteractiveObject, AIS_Line
    from OCC.Core.Prs3d import Prs3d_LineAspect
    from OCC.Core.Quantity import Quantity_Color, Quantity_NOC_WHITE, Quantity_NOC_BLACK
    from OCC.Core.Graphic3d import Graphic3d_MaterialAspect
    from OCC.Core.V3d import V3d_View, V3d_ViewProjection
    from OCC.Core.Aspect import Aspect_TOTPerspectiveProjection
    OCC_AVAILABLE = True
except ImportError:
    OCC_AVAILABLE = False
    warnings.warn("OpenCascade (pythonocc-core) not installed. Viewer will be disabled.")

from ..models.feature import CADFeature, FeatureType
from ..models.repository import FeatureRepository
from ..core.signals import bus


# DXF color index to Quantity_Color mapping (approximate)
DXF_COLOR_MAP = {
    0: Quantity_NOC_BLACK, 1: Quantity_NOC_RED, 2: Quantity_NOC_YELLOW,
    3: Quantity_NOC_GREEN, 4: Quantity_NOC_CYAN, 5: Quantity_NOC_BLUE,
    6: Quantity_NOC_MAGENTA, 7: Quantity_NOC_WHITE, 8: Quantity_NOC_BLACK,
    9: Quantity_NOC_WHITE,
}


class OCCViewerWidget(QWidget):
    """OpenCascade 3D viewer embedded in PySide6."""

    feature_clicked = Signal(str)  # feature_id when user clicks geometry

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 300)

        self._occ_context = None
        self._occ_view = None
        self._ais_objects: Dict[str, AIS_InteractiveObject] = {}  # feature_id → AIS object
        self._selection_mode = False
        self._pan_active = False
        self._last_mouse_pos: Optional[QPoint] = None

        # Initialize OCC after widget is shown (needs valid window handle)
        self._init_timer = QTimer(self)
        self._init_timer.timeout.connect(self._init_occ_viewer)
        self._init_timer.setSingleShot(True)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if OCC_AVAILABLE and self._occ_view is None:
            self._init_timer.start(100)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._occ_view:
            try:
                self._occ_view.MustBeResized()
            except Exception:
                pass

    def _init_occ_viewer(self) -> None:
        """Initialize OpenCascade display context."""
        if not OCC_AVAILABLE:
            return
        try:
            from OCC.Display.backend import load_backend
            load_backend("pyside6")
            from OCC.Display.SimpleGui import init_display
            display, start, add_menu, add_functionto_menu = init_display(self)

            self._occ_display = display
            self._occ_context = display.GetContext().GetObject()
            self._occ_view = display.GetView().GetObject()
            self._start_main_loop = start

            # Configure viewer for 2D CAD view (top-down orthographic)
            self._setup_cad_2d_view()

            # Set default selection behavior
            self._occ_context.SetSelectionModeNeutral()
            self._occ_context.SetAutoHilight(True)

            # Connect to signal bus for highlight requests
            bus.highlight_feature.connect(self._highlight_feature_by_id)
            bus.unhighlight_all.connect(self._unhighlight_all)
            bus.view_fit_all.connect(self.fit_all)
            bus.view_fit_feature.connect(self._fit_feature_by_id)

        except Exception as e:
            print(f"Failed to initialize OCC viewer: {e}")
            self._occ_view = None

    def _setup_cad_2d_view(self) -> None:
        """Configure view for 2D CAD top-down projection."""
        if not self._occ_view:
            return
        try:
            self._occ_view.SetProj(0, 0, -1)  # Top view
            self._occ_view.SetTwist(0)
            self._occ_view.SetFocalLength(1000)
            self._occ_view.SetBackFacingModel()
            self.fit_all()
        except Exception:
            pass

    def render_repository(self, repo: FeatureRepository) -> None:
        """Render all features from the repository."""
        if not self._occ_context:
            print("OCC context not ready")
            return

        # Clear existing objects
        self._clear_all_ais()

        for feature in repo.all_features():
            ais_obj = self._create_ais_from_feature(feature)
            if ais_obj:
                self._occ_context.Display(ais_obj, False)
                self._ais_objects[feature.feature_id] = ais_obj

        self._occ_context.UpdateCurrentViewer()
        self.fit_all()

    def _clear_all_ais(self) -> None:
        """Remove all AIS objects from the context."""
        if not self._occ_context:
            return
        for ais in list(self._ais_objects.values()):
            try:
                self._occ_context.Remove(ais, False)
            except Exception:
                pass
        self._ais_objects.clear()

    def _create_ais_from_feature(self, feature: CADFeature) -> Optional[AIS_InteractiveObject]:
        """Convert feature geometry dict to an AIS_Shape."""
        if not OCC_AVAILABLE:
            return None

        try:
            shape = self._create_topoDS_from_feature(feature)
            if shape is None:
                return None

            ais = AIS_Shape(shape)
            ais.SetColor(self._dxf_color_to_occ(feature.color))
            ais.SetMaterial(Graphic3d_MaterialAspect.Graphic3d_NOM_PLASTIC)
            ais.SetTransparency(0.0)

            # Set line width based on feature type
            drawer = ais.Attributes()
            line_aspect = drawer.LineAspect()
            line_aspect.SetWidth(1.0)  # Default thin line

            # Store feature_id in AIS object for selection lookup
            ais.SetOwner(feature.feature_id)

            return ais

        except Exception as e:
            print(f"Failed to create AIS for {feature.feature_type}: {e}")
            return None

    def _create_topoDS_from_feature(self, feature: CADFeature) -> Optional[TopoDS_Shape]:
        """Convert feature geometry dict to TopoDS_Shape."""
        g = feature.geometry
        ftype = feature.feature_type

        try:
            if ftype == FeatureType.LINE:
                edge = BRepBuilderAPI_MakeEdge(
                    gp_Pnt(g["x1"], g["y1"], 0), gp_Pnt(g["x2"], g["y2"], 0)
                ).Edge()
                return edge

            elif ftype == FeatureType.CIRCLE:
                circ = gp_Circ(gp_Ax2(gp_Pnt(g["cx"], g["cy"], 0), gp_Dir(0, 0, 1)), g["radius"])
                edge = BRepBuilderAPI_MakeEdge(Geom_Circle(circ)).Edge()
                return edge

            elif ftype == FeatureType.ARC:
                cx, cy, r = g["cx"], g["cy"], g["radius"]
                start_deg, end_deg = g["start_angle"], g["end_angle"]

                # Create arc of circle
                circ = gp_Circ(gp_Ax2(gp_Pnt(cx, cy, 0), gp_Dir(0, 0, 1)), r)
                arc_curve = GC_MakeArcOfCircle(circ, start_deg * math.pi / 180, end_deg * math.pi / 180, False)
                edge = BRepBuilderAPI_MakeEdge(arc_curve.Value()).Edge()
                return edge

            elif ftype == FeatureType.POLYLINE:
                points = g["points"]
                closed = g.get("closed", False)
                edges = []

                n = len(points)
                for i in range(n - 1):
                    p1 = gp_Pnt(points[i][0], points[i][1], 0)
                    p2 = gp_Pnt(points[i + 1][0], points[i + 1][1], 0)
                    edges.append(BRepBuilderAPI_MakeEdge(p1, p2).Edge())

                if closed and n > 2:
                    p1 = gp_Pnt(points[-1][0], points[-1][1], 0)
                    p2 = gp_Pnt(points[0][0], points[0][1], 0)
                    edges.append(BRepBuilderAPI_MakeEdge(p1, p2).Edge())

                wire_builder = BRepBuilderAPI_MakeWire()
                for e in edges:
                    wire_builder.Add(e)
                wire = wire_builder.Wire()
                return wire

            elif ftype == FeatureType.SPLINE:
                degree = g.get("degree", 3)
                ctrl_pts = g.get("control_points", [])
                knots = g.get("knots", [])
                rational = g.get("rational", False)

                if len(ctrl_pts) < degree + 1:
                    return None

                # Build OCC BSpline
                n_ctrl = len(ctrl_pts)
                n_knots = len(knots) if knots else n_ctrl - degree + 1

                # Create point array
                pts_array = TColgp_Array1OfPnt(1, n_ctrl)
                for i, pt in enumerate(ctrl_pts):
                    pts_array.SetValue(i + 1, gp_Pnt(pt[0], pt[1], 0))

                # Knot and multiplicity arrays
                if knots:
                    knots_array = TColStd_Array1OfReal(1, n_knots)
                    for i, k in enumerate(knots):
                        knots_array.SetValue(i + 1, k)
                else:
                    knots_array = TColStd_Array1OfReal(1, n_knots)
                    for i in range(n_knots):
                        knots_array.SetValue(i + 1, i + 1)

                mult_array = TColStd_Array1OfInteger(1, n_knots)
                for i in range(n_knots):
                    if i == 0 or i == n_knots - 1:
                        mult_array.SetValue(i + 1, degree + 1)
                    else:
                        mult_array.SetValue(i + 1, 1)

                bspline = Geom_BSplineCurve(pts_array, knots_array, mult_array, degree, rational, False, False)
                edge = BRepBuilderAPI_MakeEdge(bspline).Edge()
                return edge

            elif ftype == FeatureType.POINT:
                # Render point as a small marker
                pt = gp_Pnt(g["x"], g["y"], 0)
                # Create a tiny edge to represent the point
                tiny_edge = BRepBuilderAPI_MakeEdge(pt, gp_Pnt(g["x"] + 0.1, g["y"], 0)).Edge()
                return tiny_edge

            elif ftype in (FeatureType.TEXT, FeatureType.HATCH, FeatureType.DIMENSION):
                # Skip non-geometric annotation entities for now
                return None

            return None

        except Exception as e:
            print(f"Geometry conversion error for {ftype}: {e}")
            return None

    def _dxf_color_to_occ(self, dxf_color: int) -> Quantity_Color:
        """Map DXF ACI color index to OpenCascade Quantity_Color."""
        occ_color_name = DXF_COLOR_MAP.get(dxf_color % 256, Quantity_NOC_WHITE)
        return Quantity_Color(occ_color_name)

    # ── viewer navigation methods ──────────────────────────────────

    def fit_all(self) -> None:
        """Fit all geometry into view."""
        if self._occ_view:
            try:
                self._occ_view.FitAll(0.01)
                self._occ_view.ZFitAll()
                self._occ_view.Redraw()
            except Exception:
                pass

    def _fit_feature_by_id(self, feature_id: str) -> None:
        """Zoom to fit a specific feature."""
        ais = self._ais_objects.get(feature_id)
        if ais and self._occ_context:
            try:
                self._occ_context.Erase(ais, False)
                self._occ_context.Display(ais, True)
                # Zoom to the feature's bounding box
                bbox = ais.BoundingBox()
                self._occ_view.FitAll(bbox, 0.01)
                self._occ_view.Redraw()
            except Exception:
                pass

    def set_pan_active(self, active: bool) -> None:
        self._pan_active = active

    def set_selection_mode(self, active: bool) -> None:
        self._selection_mode = active

    # ── highlight methods (called via signal bus) ───────────────────

    def _highlight_feature_by_id(self, feature_id: str) -> None:
        """Highlight a feature visually."""
        if not self._occ_context:
            return
        ais = self._ais_objects.get(feature_id)
        if ais:
            try:
                self._occ_context.AddOrRemoveSelected(ais, False)
                self._occ_context.HilightSelected(True)
                # Set highlight properties
                drawer = ais.DynamicHilightAttributes()
                line_aspect = drawer.LineAspect()
                line_aspect.SetWidth(3.0)  # Thicker line when highlighted
                self._occ_context.UpdateCurrentViewer()
            except Exception:
                pass

    def _unhighlight_all(self) -> None:
        """Clear all highlights."""
        if not self._occ_context:
            return
        try:
            self._occ_context.ClearSelected(False)
            self._occ_context.UpdateCurrentViewer()
        except Exception:
            pass

    # ── mouse event handlers ───────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:
        super().mousePressEvent(event)
        self._last_mouse_pos = event.pos()
        if event.button() == Qt.LeftButton and self._occ_context:
            # Trigger selection at click position
            self._handle_selection(event.pos())
        elif event.button() == Qt.MiddleButton:
            self._pan_active = True

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        super().mouseReleaseEvent(event)
        self._pan_active = False
        self._last_mouse_pos = None

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        super().mouseMoveEvent(event)
        if self._pan_active and self._last_mouse_pos and self._occ_view:
            dx = event.x() - self._last_mouse_pos.x()
            dy = event.y() - self._last_mouse_pos.y()
            try:
                self._occ_view.Pan(dx, dy)
                self._occ_view.Redraw()
            except Exception:
                pass
        self._last_mouse_pos = event.pos()

    def wheelEvent(self, event: QWheelEvent) -> None:
        super().wheelEvent(event)
        if self._occ_view:
            delta = event.angleDelta().y() / 120.0
            factor = 0.9 if delta > 0 else 1.1
            try:
                self._occ_view.Zoom(factor, factor)
                self._occ_view.Redraw()
            except Exception:
                pass

    def _handle_selection(self, pos: QPoint) -> None:
        """Handle mouse click selection."""
        if not self._occ_context or not self._occ_view:
            return
        try:
            x, y = pos.x(), pos.y()
            self._occ_context.MoveTo(x, y, self._occ_view)
            self._occ_context.Select()

            selected = self._occ_context.SelectedCurrent()
            if selected:
                # Get feature_id from AIS owner
                feature_id = selected.GetOwner()
                if feature_id:
                    self.feature_clicked.emit(feature_id)
            else:
                bus.feature_deselected.emit()

        except Exception as e:
            print(f"Selection error: {e}")
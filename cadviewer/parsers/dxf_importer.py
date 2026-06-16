"""
DXFImporter — parser layer that reads DXF files via ezdxf and converts
entities into CADFeature objects stored in a FeatureRepository.

Conversion summary (DXF → OpenCascade geometry params):
  LINE      → {'x1','y1','x2','y2'}  — used later with BRepBuilderAPI_MakeEdge(Geom_Line)
  CIRCLE    → {'cx','cy','radius'}   — used later with BRepBuilderAPI_MakeEdge(Geom_Circle)
  ARC       → {'cx','cy','radius','start_angle','end_angle'} — GC_MakeArcOfCircle
  POLYLINE  → {'points':[(x,y),...], 'closed':bool} — Wire from edges
  SPLINE    → {'degree','control_points':[(x,y),...], 'knots':[], 'fit_points':[(x,y),...]}
  DIMENSION → parsed if present, stored as compound annotation
  TEXT/MTEXT → {'text','x','y','height'}

The importer does NOT depend on OpenCascade — it produces pure Python geometry
dicts that the rendering layer later converts to OCC topological objects.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import ezdxf

from ..models.feature import CADFeature, FeatureType, MeasurementMetadata, _stable_id
from ..models.repository import FeatureRepository


class DXFImporter:
    """Parse a DXF file into a FeatureRepository."""

    def __init__(self) -> None:
        self.repo = FeatureRepository()
        self._entity_index = 0
        self._insert_xscale = 1.0
        self._insert_yscale = 1.0

    def import_file(self, path: str | Path) -> FeatureRepository:
        """Load a DXF file and return populated FeatureRepository."""
        self.repo.clear()
        self._entity_index = 0
        doc = ezdxf.readfile(str(path))
        msp = doc.modelspace()

        for entity in msp:
            etype = entity.dxftype()
            if etype == "INSERT":
                self._parse_insert(entity, doc)
            else:
                self._parse_entity(entity)

        return self.repo

    def _parse_insert(self, insert_entity, doc) -> None:
        """Decompose INSERT entities (block references) into their virtual children."""
        self._insert_xscale = getattr(insert_entity.dxf, 'xscale', 1.0)
        self._insert_yscale = getattr(insert_entity.dxf, 'yscale', 1.0)
        try:
            for ve in insert_entity.virtual_entities():
                self._parse_entity(ve)
        except Exception:
            pass
        self._insert_xscale = 1.0
        self._insert_yscale = 1.0

    def _parse_entity(self, entity) -> None:
        etype = entity.dxftype()
        parser_map = {
            "LINE": self._parse_line,
            "CIRCLE": self._parse_circle,
            "ARC": self._parse_arc,
            "POLYLINE": self._parse_polyline,
            "LWPOLYLINE": self._parse_lwpolyline,
            "SPLINE": self._parse_spline,
            "ELLIPSE": self._parse_ellipse,
            "DIMENSION": self._parse_dimension,
            "MTEXT": self._parse_mtext,
            "TEXT": self._parse_text,
            "HATCH": self._parse_hatch,
            "POINT": self._parse_point,
            "SOLID": self._parse_solid,
            "LEADER": self._parse_leader,
        }
        parser = parser_map.get(etype)
        if parser:
            try:
                parser(entity)
            except Exception:
                pass

    # ── individual entity parsers ──────────────────────────────────

    def _add_line_feature(
        self,
        geom: dict,
        handle: str,
        layer: str = "0",
        color: int = 7,
    ) -> None:
        feat = CADFeature(
            feature_id=_stable_id(FeatureType.LINE, geom, handle),
            feature_type=FeatureType.LINE,
            geometry=geom,
            dxf_handle=handle,
            layer=layer,
            color=color,
        )
        self.repo.add(feat)
        self._entity_index += 1

    def _add_rectangular_polyline_edges(
        self,
        points: list,
        handle: str,
        layer: str = "0",
        color: int = 7,
    ) -> None:
        for idx in range(4):
            p1 = points[idx]
            p2 = points[(idx + 1) % 4]
            geom = {"x1": p1[0], "y1": p1[1], "x2": p2[0], "y2": p2[1]}
            edge_handle = f"{handle}:{idx + 1}" if handle else ""
            self._add_line_feature(geom, edge_handle, layer=layer, color=color)

    def _parse_line(self, e) -> None:
        s, end = e.dxf.start, e.dxf.end
        geom = {"x1": s.x, "y1": s.y, "x2": end.x, "y2": end.y}
        handle = e.dxf.handle
        self._add_line_feature(
            geom,
            handle,
            layer=e.dxf.layer if hasattr(e.dxf, "layer") else "0",
            color=e.dxf.color if hasattr(e.dxf, "color") else 7,
        )

    def _parse_circle(self, e) -> None:
        c = e.dxf.center
        cx, cy = c.x, c.y
        if self._insert_xscale < 0:
            cx = -cx
        if self._insert_yscale < 0:
            cy = -cy
        geom = {"cx": cx, "cy": cy, "radius": e.dxf.radius}
        handle = e.dxf.handle
        feat = CADFeature(
            feature_id=_stable_id(FeatureType.CIRCLE, geom, handle),
            feature_type=FeatureType.CIRCLE,
            geometry=geom,
            dxf_handle=handle,
            layer=e.dxf.layer if hasattr(e.dxf, "layer") else "0",
            color=e.dxf.color if hasattr(e.dxf, "color") else 7,
        )
        self.repo.add(feat)
        self._entity_index += 1

    def _parse_arc(self, e) -> None:
        c = e.dxf.center
        cx, cy = c.x, c.y
        start_angle = float(e.dxf.start_angle)
        end_angle = float(e.dxf.end_angle)
        # ezdxf virtual_entities() has two bugs for INSERTs with negative
        # scale: it negates ARC center coords (mirrors around origin
        # instead of insert point) and does not transform arc angles
        # for the mirror.
        if self._insert_xscale < 0 and self._insert_yscale < 0:
            cx = -cx
            cy = -cy
            start_angle += 180.0
            end_angle += 180.0
        elif self._insert_xscale < 0:
            cx = -cx
            new_start = 180.0 - end_angle
            new_end = 180.0 - start_angle
            start_angle = new_start
            end_angle = new_end
        elif self._insert_yscale < 0:
            cy = -cy
            new_start = -end_angle
            new_end = -start_angle
            start_angle = new_start
            end_angle = new_end
        geom = {
            "cx": cx, "cy": cy, "radius": e.dxf.radius,
            "start_angle": start_angle,
            "end_angle": end_angle,
        }
        handle = e.dxf.handle
        feat = CADFeature(
            feature_id=_stable_id(FeatureType.ARC, geom, handle),
            feature_type=FeatureType.ARC,
            geometry=geom,
            dxf_handle=handle,
            layer=e.dxf.layer if hasattr(e.dxf, "layer") else "0",
            color=e.dxf.color if hasattr(e.dxf, "color") else 7,
        )
        self.repo.add(feat)
        self._entity_index += 1

    def _parse_polyline(self, e) -> None:
        """Handle old-style POLYLINE (VERTEX chain) entities."""
        vertices = list(e.vertices)
        if len(vertices) < 2:
            return
        points = [(v.dxf.location.x, v.dxf.location.y) for v in vertices]
        flags = e.dxf.flags if hasattr(e.dxf, "flags") else 0
        closed = bool(flags & 1)
        layer = e.dxf.layer if hasattr(e.dxf, "layer") else "0"
        color = e.dxf.color if hasattr(e.dxf, "color") else 7
        handle = e.dxf.handle

        # Determine subtype: if only 2 vertices and not closed, treat as LINE
        if len(points) == 2 and not closed:
            geom = {"x1": points[0][0], "y1": points[0][1],
                      "x2": points[1][0], "y2": points[1][1]}
            self._add_line_feature(geom, handle, layer=layer, color=color)
        elif len(points) == 4 and closed:
            self._add_rectangular_polyline_edges(
                points, handle, layer=layer, color=color,
            )
        else:
            geom = {"points": points, "closed": closed}
            feat = CADFeature(
                feature_id=_stable_id(FeatureType.POLYLINE, geom, handle),
                feature_type=FeatureType.POLYLINE,
                geometry=geom,
                dxf_handle=handle,
                layer=layer,
                color=color,
            )
            self.repo.add(feat)
            self._entity_index += 1

    def _parse_lwpolyline(self, e) -> None:
        """Handle lightweight polyline (DXF R13+)."""
        points = [(pt[0], pt[1]) for pt in e.get_points(format="xy")]
        if len(points) < 2:
            return
        closed = e.closed
        layer = e.dxf.layer if hasattr(e.dxf, "layer") else "0"
        color = e.dxf.color if hasattr(e.dxf, "color") else 7
        handle = e.dxf.handle
        if len(points) == 4 and closed:
            self._add_rectangular_polyline_edges(
                points, handle, layer=layer, color=color,
            )
            return
        geom = {"points": points, "closed": closed}
        feat = CADFeature(
            feature_id=_stable_id(FeatureType.POLYLINE, geom, handle),
            feature_type=FeatureType.POLYLINE,
            geometry=geom,
            dxf_handle=handle,
            layer=layer,
            color=color,
        )
        self.repo.add(feat)
        self._entity_index += 1

    def _parse_spline(self, e) -> None:
        control_pts = [(p[0], p[1]) for p in e.control_points] if e.control_points else []
        fit_pts = [(p[0], p[1]) for p in e.fit_points] if e.fit_points else []
        knots = list(e.knots) if e.knots else []

        # Pre-evaluate curve using ezdxf's BSpline evaluator for accurate rendering
        eval_pts = []
        try:
            bspline = e.construction_tool()
            n_pts = max(20, len(control_pts) * 8)
            eval_pts = [(p.x, p.y) for p in bspline.approximate(n_pts)]
        except Exception:
            pass

        geom = {
            "degree": e.dxf.degree,
            "control_points": control_pts,
            "fit_points": fit_pts,
            "knots": knots,
            "eval_points": eval_pts,
            "closed": e.closed if hasattr(e, "closed") else False,
            "rational": e.dxf.flags & 4 if hasattr(e.dxf, "flags") else 0,
        }
        feat = CADFeature(
            feature_id=_stable_id(FeatureType.SPLINE, geom, e.dxf.handle),
            feature_type=FeatureType.SPLINE,
            geometry=geom,
            dxf_handle=e.dxf.handle,
            layer=e.dxf.layer if hasattr(e.dxf, "layer") else "0",
            color=e.dxf.color if hasattr(e.dxf, "color") else 7,
        )
        self.repo.add(feat)
        self._entity_index += 1

    def _parse_ellipse(self, e) -> None:
        c = e.dxf.center
        geom = {
            "cx": c.x, "cy": c.y,
            "major_axis": (e.dxf.major_axis.x, e.dxf.major_axis.y),
            "ratio": e.dxf.ratio,
            "start_param": e.dxf.start_param,
            "end_param": e.dxf.end_param,
            "is_ellipse": True,
        }
        feat = CADFeature(
            feature_id=_stable_id(FeatureType.ARC, geom, e.dxf.handle),
            feature_type=FeatureType.ARC,
            geometry=geom,
            dxf_handle=e.dxf.handle,
            layer=e.dxf.layer if hasattr(e.dxf, "layer") else "0",
            color=e.dxf.color if hasattr(e.dxf, "color") else 7,
        )
        self.repo.add(feat)
        self._entity_index += 1

    def _parse_dimension(self, e, doc=None) -> None:
        """
        Decompose DIMENSION entity into constituent geometry from its block.

        DXF dimensions store visible geometry (extension lines, arrows, measurement text)
        in an anonymous block. This parser extracts:
          - LINE entities (extension/dimension lines)
          - SOLID entities (arrow heads) → rendered as filled triangles
          - MTEXT entities (measurement value text)
        """
        # Get the dimension geometry block content
        try:
            # ezdxf provides virtual_entities() to get the block geometry
            for ve in e.virtual_entities():
                vtype = ve.dxftype()
                if vtype == "LINE":
                    self._parse_line(ve)
                elif vtype == "SOLID":
                    self._parse_solid(ve)
                elif vtype == "MTEXT":
                    self._parse_mtext(ve)
                elif vtype == "TEXT":
                    self._parse_text(ve)
        except Exception:
            pass

        # Also store the dimension entity itself for reference
        geom = {
            "dim_type": e.dxf.dimension_type if hasattr(e.dxf, "dimension_type") else 0,
            "text": e.dxf.text if hasattr(e.dxf, "text") else "",
        }
        feat = CADFeature(
            feature_id=_stable_id(FeatureType.DIMENSION, geom, e.dxf.handle),
            feature_type=FeatureType.DIMENSION,
            geometry=geom,
            dxf_handle=e.dxf.handle,
            layer=e.dxf.layer if hasattr(e.dxf, "layer") else "0",
            color=e.dxf.color if hasattr(e.dxf, "color") else 7,
        )
        self.repo.add(feat)
        self._entity_index += 1

    def _parse_solid(self, e) -> None:
        """
        Parse SOLID entity (arrow heads in dimensions).

        SOLID is a 3- or 4-point filled solid stored as vtx0..vtx3.
        """
        points = []
        for attr in ["vtx0", "vtx1", "vtx2", "vtx3"]:
            if hasattr(e.dxf, attr):
                pt = getattr(e.dxf, attr)
                if pt is not None:
                    points.append((pt.x, pt.y))

        # Deduplicate (vtx2 and vtx3 are often identical for triangles)
        if len(points) >= 3:
            # Keep unique points only for the triangle
            unique_pts = [points[0]]
            for p in points[1:]:
                if not any(abs(p[0]-u[0]) < 1e-10 and abs(p[1]-u[1]) < 1e-10 for u in unique_pts):
                    unique_pts.append(p)

            geom = {"points": unique_pts, "closed": True, "is_solid": True}
            handle = e.dxf.handle or f"_solid_{self._entity_index}"
            feat = CADFeature(
                feature_id=_stable_id(FeatureType.POLYLINE, geom, handle),
                feature_type=FeatureType.POLYLINE,
                geometry=geom,
                dxf_handle=handle,
                layer=e.dxf.layer if hasattr(e.dxf, "layer") else "0",
                color=e.dxf.color if hasattr(e.dxf, "color") else 7,
            )
            self.repo.add(feat)
            self._entity_index += 1

    def _parse_mtext(self, e) -> None:
        rotation = 0.0
        if hasattr(e.dxf, "rotation"):
            rotation = e.dxf.rotation
        elif hasattr(e.dxf, "insert"):
            pass

        geom = {
            "text": e.text,
            "x": e.dxf.insert.x, "y": e.dxf.insert.y,
            "height": e.dxf.char_height if hasattr(e.dxf, "char_height") else 2.5,
            "rotation": rotation,
        }
        feat = CADFeature(
            feature_id=_stable_id(FeatureType.TEXT, geom, e.dxf.handle),
            feature_type=FeatureType.TEXT,
            geometry=geom,
            dxf_handle=e.dxf.handle,
            layer=e.dxf.layer if hasattr(e.dxf, "layer") else "0",
            color=e.dxf.color if hasattr(e.dxf, "color") else 7,
        )
        self.repo.add(feat)
        self._entity_index += 1

    def _parse_text(self, e) -> None:
        rotation = 0.0
        if hasattr(e.dxf, "rotation"):
            rotation = e.dxf.rotation

        geom = {
            "text": e.dxf.text,
            "x": e.dxf.insert.x, "y": e.dxf.insert.y,
            "height": e.dxf.height if hasattr(e.dxf, "height") else 2.5,
            "rotation": rotation,
        }
        feat = CADFeature(
            feature_id=_stable_id(FeatureType.TEXT, geom, e.dxf.handle),
            feature_type=FeatureType.TEXT,
            geometry=geom,
            dxf_handle=e.dxf.handle,
            layer=e.dxf.layer if hasattr(e.dxf, "layer") else "0",
            color=e.dxf.color if hasattr(e.dxf, "color") else 7,
        )
        self.repo.add(feat)
        self._entity_index += 1

    def _parse_hatch(self, e) -> None:
        paths = []
        for path in e.paths:
            edges = []
            # Handle polyline-based paths
            if hasattr(path, 'vertices') and path.vertices:
                poly_pts = []
                for v in path.vertices:
                    if hasattr(v, '__getitem__') and len(v) >= 2:
                        poly_pts.append((v[0], v[1]))
                    elif hasattr(v, 'x'):
                        poly_pts.append((v.x, v.y))
                if poly_pts:
                    edges.append({"type": "Polyline", "points": poly_pts})
            else:
                for edge in path.edges:
                    edge_data = {"type": edge.EDGE_TYPE}
                    if hasattr(edge, "start") and edge.start is not None:
                        edge_data["start"] = (edge.start[0], edge.start[1])
                    if hasattr(edge, "end") and edge.end is not None:
                        edge_data["end"] = (edge.end[0], edge.end[1])
                    if hasattr(edge, "center") and edge.center is not None:
                        edge_data["center"] = (edge.center[0], edge.center[1])
                    if hasattr(edge, "radius"):
                        edge_data["radius"] = edge.radius
                    if hasattr(edge, "start_angle"):
                        edge_data["start_angle"] = edge.start_angle
                    if hasattr(edge, "end_angle"):
                        edge_data["end_angle"] = edge.end_angle
                    if hasattr(edge, "is_ccw"):
                        edge_data["is_ccw"] = edge.is_ccw
                    if hasattr(edge, "major_axis") and edge.major_axis is not None:
                        edge_data["major_axis"] = (edge.major_axis[0], edge.major_axis[1])
                    if hasattr(edge, "ratio"):
                        edge_data["ratio"] = edge.ratio
                    if hasattr(edge, "control_points") and edge.control_points:
                        edge_data["control_points"] = [(p[0], p[1]) for p in edge.control_points]
                    if hasattr(edge, "fit_points") and edge.fit_points:
                        edge_data["fit_points"] = [(p[0], p[1]) for p in edge.fit_points]
                    if hasattr(edge, "degree"):
                        edge_data["degree"] = edge.degree
                    if hasattr(edge, "knot_values") and edge.knot_values:
                        edge_data["knots"] = list(edge.knot_values)
                    # SplineEdge uses start_point/end_point instead of start/end
                    if "start" not in edge_data and hasattr(edge, "start_point"):
                        try:
                            sp = edge.start_point
                            edge_data["start"] = (sp[0], sp[1])
                        except Exception:
                            pass
                    if "end" not in edge_data and hasattr(edge, "end_point"):
                        try:
                            ep = edge.end_point
                            edge_data["end"] = (ep[0], ep[1])
                        except Exception:
                            pass
                    edges.append(edge_data)
            paths.append(edges)

        feat = CADFeature(
            feature_id=_stable_id(FeatureType.HATCH, {"paths": len(paths), "pattern": e.dxf.pattern_name if hasattr(e.dxf, "pattern_name") else ""}, e.dxf.handle),
            feature_type=FeatureType.HATCH,
            geometry={"paths": paths, "pattern": e.dxf.pattern_name if hasattr(e.dxf, "pattern_name") else ""},
            dxf_handle=e.dxf.handle,
            layer=e.dxf.layer if hasattr(e.dxf, "layer") else "0",
            color=e.dxf.color if hasattr(e.dxf, "color") else 7,
        )
        self.repo.add(feat)
        self._entity_index += 1

    def _parse_point(self, e) -> None:
        loc = e.dxf.location
        geom = {"x": loc.x, "y": loc.y}
        feat = CADFeature(
            feature_id=_stable_id(FeatureType.POINT, geom, e.dxf.handle),
            feature_type=FeatureType.POINT,
            geometry=geom,
            dxf_handle=e.dxf.handle,
            layer=e.dxf.layer if hasattr(e.dxf, "layer") else "0",
            color=e.dxf.color if hasattr(e.dxf, "color") else 7,
        )
        self.repo.add(feat)
        self._entity_index += 1

    def _parse_leader(self, e) -> None:
        """Parse LEADER entity as a polyline with arrowhead."""
        vertices = []
        for v in e.vertices:
            if hasattr(v, 'x'):
                vertices.append((v.x, v.y))
            else:
                vertices.append((v[0], v[1]))
        if len(vertices) < 2:
            return

        geom = {"points": vertices, "closed": False}
        feat = CADFeature(
            feature_id=_stable_id(FeatureType.LEADER, geom, e.dxf.handle),
            feature_type=FeatureType.LEADER,
            geometry=geom,
            dxf_handle=e.dxf.handle,
            layer=e.dxf.layer if hasattr(e.dxf, "layer") else "0",
            color=e.dxf.color if hasattr(e.dxf, "color") else 7,
        )
        self.repo.add(feat)
        self._entity_index += 1

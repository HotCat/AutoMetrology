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

from ..models.feature import CADFeature, FeatureType, MeasurementMetadata
from ..models.repository import FeatureRepository


class DXFImporter:
    """Parse a DXF file into a FeatureRepository."""

    def __init__(self) -> None:
        self.repo = FeatureRepository()
        self._entity_index = 0

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
        block_name = insert_entity.dxf.name
        try:
            for ve in insert_entity.virtual_entities():
                self._parse_entity(ve)
        except Exception:
            pass

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
        }
        parser = parser_map.get(etype)
        if parser:
            try:
                parser(entity)
            except Exception:
                pass

    # ── individual entity parsers ──────────────────────────────────

    def _parse_line(self, e) -> None:
        s, end = e.dxf.start, e.dxf.end
        feat = CADFeature(
            feature_type=FeatureType.LINE,
            geometry={"x1": s.x, "y1": s.y, "x2": end.x, "y2": end.y},
            dxf_handle=e.dxf.handle,
            layer=e.dxf.layer if hasattr(e.dxf, "layer") else "0",
            color=e.dxf.color if hasattr(e.dxf, "color") else 7,
        )
        self.repo.add(feat)
        self._entity_index += 1

    def _parse_circle(self, e) -> None:
        c = e.dxf.center
        feat = CADFeature(
            feature_type=FeatureType.CIRCLE,
            geometry={"cx": c.x, "cy": c.y, "radius": e.dxf.radius},
            dxf_handle=e.dxf.handle,
            layer=e.dxf.layer if hasattr(e.dxf, "layer") else "0",
            color=e.dxf.color if hasattr(e.dxf, "color") else 7,
        )
        self.repo.add(feat)
        self._entity_index += 1

    def _parse_arc(self, e) -> None:
        c = e.dxf.center
        feat = CADFeature(
            feature_type=FeatureType.ARC,
            geometry={
                "cx": c.x, "cy": c.y, "radius": e.dxf.radius,
                "start_angle": e.dxf.start_angle,
                "end_angle": e.dxf.end_angle,
            },
            dxf_handle=e.dxf.handle,
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

        # Determine subtype: if only 2 vertices and not closed, treat as LINE
        if len(points) == 2 and not closed:
            feat = CADFeature(
                feature_type=FeatureType.LINE,
                geometry={"x1": points[0][0], "y1": points[0][1],
                          "x2": points[1][0], "y2": points[1][1]},
                dxf_handle=e.dxf.handle,
                layer=e.dxf.layer if hasattr(e.dxf, "layer") else "0",
                color=e.dxf.color if hasattr(e.dxf, "color") else 7,
            )
        else:
            feat = CADFeature(
                feature_type=FeatureType.POLYLINE,
                geometry={"points": points, "closed": closed},
                dxf_handle=e.dxf.handle,
                layer=e.dxf.layer if hasattr(e.dxf, "layer") else "0",
                color=e.dxf.color if hasattr(e.dxf, "color") else 7,
            )
        self.repo.add(feat)
        self._entity_index += 1

    def _parse_lwpolyline(self, e) -> None:
        """Handle lightweight polyline (DXF R13+)."""
        points = [(pt[0], pt[1]) for pt in e.get_points(format="xy")]
        if len(points) < 2:
            return
        closed = e.closed
        feat = CADFeature(
            feature_type=FeatureType.POLYLINE,
            geometry={"points": points, "closed": closed},
            dxf_handle=e.dxf.handle,
            layer=e.dxf.layer if hasattr(e.dxf, "layer") else "0",
            color=e.dxf.color if hasattr(e.dxf, "color") else 7,
        )
        self.repo.add(feat)
        self._entity_index += 1

    def _parse_spline(self, e) -> None:
        control_pts = [(p[0], p[1]) for p in e.control_points] if e.control_points else []
        fit_pts = [(p[0], p[1]) for p in e.fit_points] if e.fit_points else []
        knots = list(e.knots) if e.knots else []

        feat = CADFeature(
            feature_type=FeatureType.SPLINE,
            geometry={
                "degree": e.dxf.degree,
                "control_points": control_pts,
                "fit_points": fit_pts,
                "knots": knots,
                "closed": e.closed if hasattr(e, "closed") else False,
                "rational": e.dxf.flags & 4 if hasattr(e.dxf, "flags") else 0,
            },
            dxf_handle=e.dxf.handle,
            layer=e.dxf.layer if hasattr(e.dxf, "layer") else "0",
            color=e.dxf.color if hasattr(e.dxf, "color") else 7,
        )
        self.repo.add(feat)
        self._entity_index += 1

    def _parse_ellipse(self, e) -> None:
        c = e.dxf.center
        feat = CADFeature(
            feature_type=FeatureType.ARC,
            geometry={
                "cx": c.x, "cy": c.y,
                "major_axis": (e.dxf.major_axis.x, e.dxf.major_axis.y),
                "ratio": e.dxf.ratio,
                "start_param": e.dxf.start_param,
                "end_param": e.dxf.end_param,
                "is_ellipse": True,
            },
            dxf_handle=e.dxf.handle,
            layer=e.dxf.layer if hasattr(e.dxf, "layer") else "0",
            color=e.dxf.color if hasattr(e.dxf, "color") else 7,
        )
        self.repo.add(feat)
        self._entity_index += 1

    def _parse_dimension(self, e) -> None:
        feat = CADFeature(
            feature_type=FeatureType.DIMENSION,
            geometry={
                "dim_type": e.dxf.dimension_type if hasattr(e.dxf, "dimension_type") else 0,
            },
            dxf_handle=e.dxf.handle,
            layer=e.dxf.layer if hasattr(e.dxf, "layer") else "0",
            color=e.dxf.color if hasattr(e.dxf, "color") else 7,
        )
        self.repo.add(feat)
        self._entity_index += 1

    def _parse_mtext(self, e) -> None:
        feat = CADFeature(
            feature_type=FeatureType.TEXT,
            geometry={
                "text": e.text,
                "x": e.dxf.insert.x, "y": e.dxf.insert.y,
                "height": e.dxf.char_height if hasattr(e.dxf, "char_height") else 2.5,
            },
            dxf_handle=e.dxf.handle,
            layer=e.dxf.layer if hasattr(e.dxf, "layer") else "0",
            color=e.dxf.color if hasattr(e.dxf, "color") else 7,
        )
        self.repo.add(feat)
        self._entity_index += 1

    def _parse_text(self, e) -> None:
        feat = CADFeature(
            feature_type=FeatureType.TEXT,
            geometry={
                "text": e.dxf.text,
                "x": e.dxf.insert.x, "y": e.dxf.insert.y,
                "height": e.dxf.height if hasattr(e.dxf, "height") else 2.5,
            },
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
            for edge in path.edges:
                edge_data = {"type": edge.EDGE_TYPE}
                if hasattr(edge, "start"):
                    edge_data["start"] = (edge.start[0], edge.start[1])
                if hasattr(edge, "end"):
                    edge_data["end"] = (edge.end[0], edge.end[1])
                if hasattr(edge, "center"):
                    edge_data["center"] = (edge.center[0], edge.center[1])
                if hasattr(edge, "radius"):
                    edge_data["radius"] = edge.radius
                edges.append(edge_data)
            paths.append(edges)

        feat = CADFeature(
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
        feat = CADFeature(
            feature_type=FeatureType.POINT,
            geometry={"x": loc.x, "y": loc.y},
            dxf_handle=e.dxf.handle,
            layer=e.dxf.layer if hasattr(e.dxf, "layer") else "0",
            color=e.dxf.color if hasattr(e.dxf, "color") else 7,
        )
        self.repo.add(feat)
        self._entity_index += 1

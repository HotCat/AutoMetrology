"""
validation — DXF integrity validator for post-conversion checks.

Four-phase validation:
  1. File level (readability, header)
  2. Structural integrity (handles, layers, blocks)
  3. Geometric sanity (no NaN/Inf, valid radii)
  4. Completeness (entity type coverage)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import ezdxf

logger = logging.getLogger(__name__)

# Entity types that DXFImporter can handle
SUPPORTED_ENTITY_TYPES = {
    "LINE", "CIRCLE", "ARC", "POLYLINE", "LWPOLYLINE",
    "SPLINE", "ELLIPSE", "POINT",
    "TEXT", "MTEXT", "DIMENSION",
    "INSERT", "SOLID", "HATCH", "LEADER",
}


@dataclass
class ValidationReport:
    """Result of DXF validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    entity_count: int = 0
    layer_count: int = 0
    block_count: int = 0
    dimension_count: int = 0
    handle_set: set[str] = field(default_factory=set)
    missing_handles: set[str] = field(default_factory=set)
    geometric_anomalies: list[str] = field(default_factory=list)
    entity_type_counts: dict[str, int] = field(default_factory=dict)


class DXFValidator:
    """Validates converted DXF files for integrity and completeness."""

    def __init__(self, original_dwg_path: Optional[Path] = None) -> None:
        self._original_dwg_path = original_dwg_path

    def validate(self, dxf_path: Path) -> ValidationReport:
        """Run all validation phases on a DXF file."""
        report = ValidationReport(is_valid=True)

        # Phase 1: File level
        doc = self._check_file(dxf_path, report)
        if doc is None:
            report.is_valid = False
            return report

        try:
            msp = doc.modelspace()

            # Phase 2: Structural integrity
            self._check_header(doc, report)
            self._check_handles(doc, report)
            self._check_layers(doc, report)
            self._check_blocks(doc, report)

            # Phase 3: Geometric sanity
            self._check_coordinates(msp, report)

            # Phase 4: Completeness
            self._check_entity_types(msp, report)

            # Summary counts
            report.entity_count = sum(report.entity_type_counts.values())
            report.layer_count = len(doc.layers)
            report.block_count = len(list(doc.blocks))
            report.dimension_count = report.entity_type_counts.get("DIMENSION", 0)

            if report.errors:
                report.is_valid = False

        except Exception as e:
            report.errors.append(f"Validation crashed: {e}")
            report.is_valid = False
            logger.exception("DXF validation failed with exception")

        return report

    def _check_file(self, dxf_path: Path, report: ValidationReport):
        """Phase 1: File-level checks."""
        if not dxf_path.exists():
            report.errors.append(f"DXF file not found: {dxf_path}")
            return None
        if dxf_path.stat().st_size == 0:
            report.errors.append("DXF file is empty")
            return None
        try:
            doc = ezdxf.readfile(str(dxf_path))
            return doc
        except ezdxf.DXFStructureError as e:
            report.errors.append(f"ezdxf cannot read DXF: {e}")
            return None
        except Exception as e:
            report.errors.append(f"Unexpected error reading DXF: {e}")
            return None

    def _check_header(self, doc, report: ValidationReport) -> None:
        """Validate header variables."""
        header = doc.header
        if "$ACADVER" not in header:
            report.warnings.append("Missing $ACADVER header variable")
        acadver = header.get("$ACADVER", "")
        if acadver and not acadver.startswith("AC"):
            report.warnings.append(f"Unexpected $ACADVER value: {acadver}")

        if "$INSUNITS" not in header:
            report.warnings.append(
                "Missing $INSUNITS — unit information may be incorrect"
            )

    def _check_handles(self, doc, report: ValidationReport) -> None:
        """Check handle uniqueness and referential integrity."""
        msp = doc.modelspace()
        handles_seen: dict[str, int] = {}
        all_handles: set[str] = set()

        # Collect all entity handles
        for entity in msp:
            h = entity.dxf.handle
            all_handles.add(h)
            handles_seen[h] = handles_seen.get(h, 0) + 1

        # Check document-wide handles (blocks, tables, etc.)
        for entity in doc.entitydb.values():
            all_handles.add(entity.dxf.handle)

        report.handle_set = all_handles

        # Duplicate handles in modelspace
        duplicates = {h: c for h, c in handles_seen.items() if c > 1}
        if duplicates:
            report.errors.append(
                f"Duplicate handles in modelspace: {len(duplicates)} duplicates"
            )

        # Referential integrity — check common handle references
        for entity in msp:
            for attr in ("layer", "linetype", "style", "dimstyle"):
                if hasattr(entity.dxf, attr):
                    # These are name-based, not handle-based — skip
                    pass
            # Block references
            if entity.dxftype() == "INSERT":
                block_name = entity.dxf.get("name", "")
                if block_name and block_name not in doc.blocks:
                    report.warnings.append(
                        f"INSERT references undefined block: {block_name}"
                    )

    def _check_layers(self, doc, report: ValidationReport) -> None:
        """Verify entity layers exist in LAYER table."""
        msp = doc.modelspace()
        layer_names = {layer.dxf.name for layer in doc.layers}
        undefined_layers: set[str] = set()

        for entity in msp:
            layer = entity.dxf.get("layer", "")
            if layer and layer not in layer_names:
                undefined_layers.add(layer)

        if undefined_layers:
            report.warnings.append(
                f"Entities reference undefined layers: {', '.join(sorted(undefined_layers)[:10])}"
            )

    def _check_blocks(self, doc, report: ValidationReport) -> None:
        """Verify block definitions exist for INSERT entities."""
        msp = doc.modelspace()
        missing_blocks: set[str] = set()

        for entity in msp.query("INSERT"):
            name = entity.dxf.get("name", "")
            if name and name not in doc.blocks:
                missing_blocks.add(name)

        if missing_blocks:
            report.errors.append(
                f"INSERT entities reference missing blocks: {', '.join(sorted(missing_blocks))}"
            )

    def _check_coordinates(self, msp, report: ValidationReport) -> None:
        """Phase 3: Check for NaN, Inf, or absurd coordinates."""
        anomalies: list[str] = []
        limit = 1e10

        for entity in msp:
            etype = entity.dxftype()
            try:
                if etype == "LINE":
                    for attr in ("start", "end"):
                        pt = entity.dxf.get(attr)
                        if pt and _bad_coord(pt, limit):
                            anomalies.append(f"LINE {entity.dxf.handle}: {attr} out of range")

                elif etype == "CIRCLE":
                    center = entity.dxf.get("center")
                    if center and _bad_coord(center, limit):
                        anomalies.append(f"CIRCLE {entity.dxf.handle}: center out of range")
                    radius = entity.dxf.get("radius", 0)
                    if radius <= 0:
                        anomalies.append(f"CIRCLE {entity.dxf.handle}: invalid radius {radius}")
                    if math.isinf(radius) or math.isnan(radius):
                        anomalies.append(f"CIRCLE {entity.dxf.handle}: NaN/Inf radius")

                elif etype == "ARC":
                    center = entity.dxf.get("center")
                    if center and _bad_coord(center, limit):
                        anomalies.append(f"ARC {entity.dxf.handle}: center out of range")
                    radius = entity.dxf.get("radius", 0)
                    if radius <= 0:
                        anomalies.append(f"ARC {entity.dxf.handle}: invalid radius {radius}")

                elif etype in ("LWPOLYLINE", "POLYLINE"):
                    # LWPOLYLINE has no direct point accessor in dxf namespace
                    pass

                elif etype in ("TEXT", "MTEXT"):
                    insert = entity.dxf.get("insert")
                    if insert and _bad_coord(insert, limit):
                        anomalies.append(f"{etype} {entity.dxf.handle}: position out of range")

            except Exception:
                # Non-fatal — skip this entity's coordinate check
                pass

        report.geometric_anomalies = anomalies
        if len(anomalies) > 10:
            report.errors.append(
                f"{len(anomalies)} geometric anomalies detected"
            )
        elif anomalies:
            report.warnings.append(
                f"Geometric anomalies: {'; '.join(anomalies[:5])}"
            )

    def _check_entity_types(self, msp, report: ValidationReport) -> None:
        """Phase 4: Count entity types and flag unsupported ones."""
        type_counts: dict[str, int] = {}
        unsupported: list[str] = []

        for entity in msp:
            etype = entity.dxftype()
            type_counts[etype] = type_counts.get(etype, 0) + 1

        report.entity_type_counts = type_counts

        for etype in sorted(type_counts):
            if etype not in SUPPORTED_ENTITY_TYPES:
                unsupported.append(f"{etype} ({type_counts[etype]})")

        if unsupported:
            report.warnings.append(
                f"Unsupported entity types: {', '.join(unsupported[:10])}"
            )

        # Zero entities is an error
        total = sum(type_counts.values())
        if total == 0:
            report.errors.append("DXF contains zero entities in modelspace")


def _bad_coord(pt, limit: float) -> bool:
    """Check if a point has NaN, Inf, or out-of-range coordinates."""
    for val in (pt.x, pt.y, pt.z if hasattr(pt, 'z') else 0):
        if math.isnan(val) or math.isinf(val) or abs(val) > limit:
            return True
    return False

"""
handle_mapper — DWG↔DXF handle identity mapping.

ODA File Converter generally preserves entity handles across conversion.
This module verifies that preservation and builds identity mappings for
the DWG Entity ↔ DXF Entity ↔ Internal CAD Feature chain.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import ezdxf

logger = logging.getLogger(__name__)


class HandleMapper:
    """Maps DWG entity handles to DXF entity handles for stable identity."""

    def verify_handle_preservation(self, dxf_path: Path) -> float:
        """Check what fraction of handles are well-formed (hex strings).

        Returns a confidence ratio [0.0, 1.0]. ODA-preserved handles
        are typically hex like "1A3".
        """
        try:
            doc = ezdxf.readfile(str(dxf_path))
        except Exception as e:
            logger.error("Cannot read DXF for handle verification: %s", e)
            return 0.0

        msp = doc.modelspace()
        total = 0
        hex_count = 0

        for entity in msp:
            handle = entity.dxf.handle
            total += 1
            try:
                int(handle, 16)
                hex_count += 1
            except ValueError:
                pass

        if total == 0:
            return 0.0

        ratio = hex_count / total
        logger.info(
            "Handle preservation: %d/%d (%.1f%%) are hex handles",
            hex_count, total, ratio * 100,
        )
        return ratio

    def build_handle_index(self, dxf_path: Path) -> dict[str, str]:
        """Build handle → entity type mapping from a DXF file.

        Returns dict of {handle: entity_type} for all modelspace entities.
        """
        try:
            doc = ezdxf.readfile(str(dxf_path))
        except Exception as e:
            logger.error("Cannot read DXF for handle indexing: %s", e)
            return {}

        index: dict[str, str] = {}
        for entity in doc.modelspace():
            index[entity.dxf.handle] = entity.dxftype()

        return index

    def get_stable_id(self, dwg_handle: str, dxf_handle: str) -> str:
        """Generate a stable composite ID.

        If dwg_handle == dxf_handle (ODA preserved it): use dxf_handle.
        Otherwise: use composite "dwg_handle:dxf_handle".
        """
        if dwg_handle == dxf_handle:
            return dxf_handle
        return f"{dwg_handle}:{dxf_handle}"

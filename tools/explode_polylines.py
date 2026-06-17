#!/usr/bin/env python3
"""Convert DXF polyline segments into LINE entities."""

from __future__ import annotations

import sys

from explode_rectangular_polylines import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

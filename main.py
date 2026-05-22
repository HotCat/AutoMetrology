#!/usr/bin/env python3
"""
CAD Inspection Tool — application entry point.

Usage:
    python main.py [dxf_or_dwg_file]

If a DXF file path is provided, it loads directly.
If a DWG file path is provided, it converts to DXF first, then loads.
"""

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from cadviewer.ui.main_window import MainWindow


def main() -> int:
    # Enable high-DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("CAD Inspection Tool")
    app.setOrganizationName("Metrology")

    window = MainWindow()
    window.show()

    # Auto-load file if provided as command-line argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if not Path(file_path).exists():
            print(f"File not found: {file_path}")
            return 1

        lower = file_path.lower()
        if lower.endswith(".dxf"):
            from PySide6.QtCore import QTimer
            QTimer.singleShot(500, lambda: window._load_dxf(file_path))
        elif lower.endswith(".dwg"):
            from PySide6.QtCore import QTimer
            QTimer.singleShot(500, lambda: window._open_dwg_path(file_path))
        else:
            print(f"Unsupported file format: {file_path}")

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

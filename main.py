#!/usr/bin/env python3
"""
CAD Inspection Tool — application entry point.

Usage:
    python main.py [dxf_file]

If a DXF file path is provided as argument, it will be loaded on startup.
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

    # Auto-load DXF if provided as command-line argument
    if len(sys.argv) > 1:
        dxf_path = sys.argv[1]
        if Path(dxf_path).exists():
            # Delay loading until event loop starts
            from PySide6.QtCore import QTimer
            QTimer.singleShot(500, lambda: window._load_dxf(dxf_path))

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

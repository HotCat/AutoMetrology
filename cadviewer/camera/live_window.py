"""
CameraLiveWindow — dedicated full-size live preview window for focus adjustment.

Contains the camera settings panel as a collapsible right sidebar so the user
can adjust exposure, gain, etc. while watching the live feed at full size.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSplitter, QGroupBox,
)

from .preview_widget import CameraPreviewWidget
from .settings_widget import CameraSettingsWidget


class CameraLiveWindow(QWidget):
    """Full-size live preview sub-window for camera focus adjustment."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Camera Live Preview")
        self.setWindowFlags(Qt.Window | Qt.WindowMinMaxButtonsHint)
        self.resize(1280, 800)
        self._latest_frame: np.ndarray | None = None

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)

        # ── Left: live preview ──────────────────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self._preview = CameraPreviewWidget()
        self._preview._label.setMinimumHeight(200)
        left_layout.addWidget(self._preview, stretch=1)

        # Bottom toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(6, 4, 6, 4)
        toolbar.setSpacing(6)

        self._btn_capture = QPushButton("Capture Frame")
        self._btn_capture.setStyleSheet("""
            QPushButton {
                background: #264f78; color: white; border: none;
                padding: 6px 16px; border-radius: 3px; font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background: #306898; }
        """)
        toolbar.addWidget(self._btn_capture)

        self._btn_fit = QPushButton("Fit to Window")
        self._btn_fit.setStyleSheet("""
            QPushButton {
                background: #333; color: #ccc; border: 1px solid #555;
                padding: 6px 12px; border-radius: 3px; font-size: 12px;
            }
            QPushButton:hover { background: #444; }
        """)
        self._btn_fit.clicked.connect(self._fit_to_window)
        toolbar.addWidget(self._btn_fit)

        self._status = QLabel("Waiting for frames...")
        self._status.setStyleSheet("color: #888; font-size: 10px;")
        toolbar.addWidget(self._status)
        toolbar.addStretch()

        self._resolution_label = QLabel("")
        self._resolution_label.setStyleSheet("color: #666; font-size: 10px;")
        toolbar.addWidget(self._resolution_label)

        left_layout.addLayout(toolbar)
        splitter.addWidget(left)

        # ── Right: settings sidebar ─────────────────────────────────
        right = QWidget()
        right.setMaximumWidth(320)
        right.setMinimumWidth(200)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)

        settings_header = QLabel("Camera Settings")
        settings_header.setStyleSheet(
            "font-weight: bold; padding: 4px; background: #2d2d2d; color: #ddd;"
        )
        right_layout.addWidget(settings_header)

        self._settings = CameraSettingsWidget()
        right_layout.addWidget(self._settings)

        right_layout.addStretch()
        splitter.addWidget(right)

        # Initial split: 75% preview, 25% settings
        splitter.setSizes([960, 320])

        outer.addWidget(splitter)

    @property
    def settings_widget(self) -> CameraSettingsWidget:
        """Expose the embedded settings widget for external wiring."""
        return self._settings

    def display_frame(self, frame: np.ndarray) -> None:
        """Receive and display a live frame from the camera."""
        self._latest_frame = frame
        self._preview.display_frame(frame)

        h, w = frame.shape[:2]
        self._resolution_label.setText(f"{w}x{h}")
        self._status.setText("Live")

    def get_latest_frame(self) -> np.ndarray | None:
        return self._latest_frame

    def clear(self) -> None:
        """Reset to placeholder state when camera closes."""
        self._latest_frame = None
        self._preview.set_placeholder_text("Camera closed")
        self._status.setText("Camera closed")
        self._resolution_label.setText("")

    def _fit_to_window(self) -> None:
        if self._latest_frame is not None:
            self._preview.display_frame(self._latest_frame)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_F:
            self._fit_to_window()
        else:
            super().keyPressEvent(event)

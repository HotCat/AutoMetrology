"""
CameraLiveWindow — dedicated full-size live preview window for focus adjustment.

Contains the camera settings panel as a collapsible right sidebar so the user
can adjust exposure, gain, etc. while watching the live feed at full size.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSplitter, QSlider, QSpinBox,
)

from .preview_widget import CameraPreviewWidget
from .settings_widget import CameraSettingsWidget
from ..ui.light_control_panel import LightControlPanel


class CameraLiveWindow(QWidget):
    """Full-size live preview sub-window for camera focus adjustment."""

    closed = Signal()

    def __init__(self, config=None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Camera Live Preview")
        self.setWindowFlags(Qt.Window | Qt.WindowMinMaxButtonsHint)
        self.resize(1280, 800)
        self._latest_frame: np.ndarray | None = None
        self._config = config

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

        toolbar.addWidget(QLabel("Zoom:"))
        self._btn_zoom_out = QPushButton("-")
        self._btn_zoom_out.setFixedWidth(32)
        self._btn_zoom_out.clicked.connect(self._zoom_out)
        toolbar.addWidget(self._btn_zoom_out)

        self._zoom_slider = QSlider(Qt.Horizontal)
        self._zoom_slider.setRange(10, 2000)
        self._zoom_slider.setSingleStep(10)
        self._zoom_slider.setPageStep(100)
        self._zoom_slider.setValue(100)
        self._zoom_slider.setMinimumWidth(180)
        self._zoom_slider.valueChanged.connect(self._on_zoom_value_changed)
        toolbar.addWidget(self._zoom_slider)

        self._zoom_spin = QSpinBox()
        self._zoom_spin.setRange(10, 2000)
        self._zoom_spin.setSingleStep(10)
        self._zoom_spin.setSuffix("%")
        self._zoom_spin.setValue(100)
        self._zoom_spin.valueChanged.connect(self._on_zoom_value_changed)
        toolbar.addWidget(self._zoom_spin)

        self._btn_zoom_in = QPushButton("+")
        self._btn_zoom_in.setFixedWidth(32)
        self._btn_zoom_in.clicked.connect(self._zoom_in)
        toolbar.addWidget(self._btn_zoom_in)

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

        self._light_panel = None
        if self._config is not None:
            self._light_panel = LightControlPanel(self._config)
            right_layout.addWidget(self._light_panel)

        right_layout.addStretch()
        splitter.addWidget(right)

        # Initial split: 75% preview, 25% settings
        splitter.setSizes([960, 320])

        outer.addWidget(splitter)

    @property
    def settings_widget(self) -> CameraSettingsWidget:
        """Expose the embedded settings widget for external wiring."""
        return self._settings

    @property
    def light_panel(self):
        """Expose the embedded light panel for cleanup/testing."""
        return self._light_panel

    def display_frame(self, frame: np.ndarray) -> None:
        """Receive and display a live frame from the camera."""
        self._latest_frame = frame
        self._preview.display_frame(frame)

        h, w = frame.shape[:2]
        self._resolution_label.setText(f"{w}x{h}")
        zoom = self._preview.zoom_percent
        self._status.setText("Live Fit" if zoom is None else f"Live {zoom:.0f}%")

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
            self._preview.fit_to_window()
            self._status.setText("Live Fit")

    def _on_zoom_value_changed(self, value: int) -> None:
        sender = self.sender()
        if sender is not self._zoom_slider:
            self._zoom_slider.blockSignals(True)
            self._zoom_slider.setValue(value)
            self._zoom_slider.blockSignals(False)
        if sender is not self._zoom_spin:
            self._zoom_spin.blockSignals(True)
            self._zoom_spin.setValue(value)
            self._zoom_spin.blockSignals(False)
        self._preview.set_zoom_percent(float(value))
        self._status.setText(f"Live {value}%")

    def _set_zoom_control_value(self, value: float) -> None:
        value_i = int(np.clip(round(value), 10, 2000))
        self._zoom_slider.blockSignals(True)
        self._zoom_spin.blockSignals(True)
        self._zoom_slider.setValue(value_i)
        self._zoom_spin.setValue(value_i)
        self._zoom_slider.blockSignals(False)
        self._zoom_spin.blockSignals(False)

    def _zoom_in(self) -> None:
        self._preview.zoom_in()
        zoom = self._preview.zoom_percent or 100.0
        self._set_zoom_control_value(zoom)
        self._status.setText(f"Live {zoom:.0f}%")

    def _zoom_out(self) -> None:
        self._preview.zoom_out()
        zoom = self._preview.zoom_percent or 100.0
        self._set_zoom_control_value(zoom)
        self._status.setText(f"Live {zoom:.0f}%")

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_F:
            self._fit_to_window()
        elif event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self._zoom_in()
        elif event.key() == Qt.Key_Minus:
            self._zoom_out()
        elif event.key() == Qt.Key_0:
            self._set_zoom_control_value(100)
            self._preview.set_zoom_percent(100.0)
            self._status.setText("Live 100%")
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        if self._light_panel is not None:
            self._light_panel.close_controller()
        self.closed.emit()
        super().closeEvent(event)

"""
CameraPreviewWidget — displays camera live feed frames as a QLabel.

Receives BGR numpy arrays via display_frame() slot, converts to QPixmap,
and scales to widget size while maintaining aspect ratio.
"""

from __future__ import annotations

import time
import numpy as np
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QWidget, QVBoxLayout

try:
    import cv2
except ImportError:  # pragma: no cover - optional runtime dependency
    cv2 = None


class CameraPreviewWidget(QWidget):
    """Embeddable camera preview display."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._latest_frame: np.ndarray | None = None
        self._frame_counter: int = 0
        self._latest_frame_time: float = 0.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel("No camera")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setMinimumHeight(120)
        self._label.setStyleSheet("""
            QLabel {
                background-color: #111;
                color: #555;
                font-size: 12px;
                border: 1px solid #333;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self._label)

    def display_frame(self, frame: np.ndarray) -> None:
        """Display a BGR numpy frame, scaled to widget size."""
        self._latest_frame = frame
        self._frame_counter += 1
        self._latest_frame_time = time.monotonic()
        self._render_frame(frame)

    def _render_frame(self, frame: np.ndarray) -> None:
        if frame is None:
            return

        frame = self._frame_for_display(frame)
        h, w = frame.shape[:2]
        if len(frame.shape) == 2:
            # Grayscale
            qimg = QImage(frame.data, w, h, w, QImage.Format_Grayscale8).copy()
        elif frame.shape[2] == 1:
            qimg = QImage(frame.data, w, h, w, QImage.Format_Grayscale8).copy()
        else:
            # BGR → RGB
            rgb = frame[:, :, ::-1].copy()
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()

        pixmap = QPixmap.fromImage(qimg)
        self._label.setPixmap(
            pixmap.scaled(
                self._label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def set_placeholder_text(self, text: str) -> None:
        """Show placeholder text when no camera is active."""
        self._label.clear()
        self._label.setText(text)

    def get_latest_frame(self) -> np.ndarray | None:
        """Return the most recently displayed frame."""
        return self._latest_frame

    @property
    def frame_counter(self) -> int:
        return self._frame_counter

    @property
    def latest_frame_age_s(self) -> float:
        if self._latest_frame_time <= 0.0:
            return float("inf")
        return max(0.0, time.monotonic() - self._latest_frame_time)

    def resizeEvent(self, event) -> None:
        """Re-scale pixmap on resize."""
        super().resizeEvent(event)
        if self._latest_frame is not None:
            self._render_frame(self._latest_frame)

    def _frame_for_display(self, frame: np.ndarray) -> np.ndarray:
        """Downsample large frames before UI-thread color conversion."""
        label_size = self._label.size()
        max_w = max(1, int(label_size.width()))
        max_h = max(1, int(label_size.height()))
        h, w = frame.shape[:2]
        scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
        if scale >= 1.0:
            return frame

        out_w = max(1, int(w * scale))
        out_h = max(1, int(h * scale))
        if cv2 is not None:
            return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        y_idx = np.linspace(0, h - 1, out_h).astype(np.intp)
        x_idx = np.linspace(0, w - 1, out_w).astype(np.intp)
        return frame[np.ix_(y_idx, x_idx)]

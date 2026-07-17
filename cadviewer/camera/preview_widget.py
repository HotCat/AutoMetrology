"""
CameraPreviewWidget — displays camera live feed frames as a QLabel.

Receives BGR numpy arrays via display_frame() slot, converts to QPixmap,
and scales to widget size while maintaining aspect ratio.
"""

from __future__ import annotations

import time
import numpy as np
from PySide6.QtCore import Qt, QPointF, QSize
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QWidget, QVBoxLayout, QSizePolicy

try:
    import cv2
except ImportError:  # pragma: no cover - optional runtime dependency
    cv2 = None


class _PreviewLabel(QLabel):
    def __init__(self, owner: "CameraPreviewWidget") -> None:
        super().__init__("No camera")
        self._owner = owner
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

    def sizeHint(self) -> QSize:
        return QSize(640, 480)

    def minimumSizeHint(self) -> QSize:
        return QSize(160, 120)

    def wheelEvent(self, event) -> None:
        self._owner._handle_wheel_zoom(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._owner._start_pan(event.position())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._owner._update_pan(event.position()):
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._owner._end_pan()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        self._owner.fit_to_window()
        event.accept()


class CameraPreviewWidget(QWidget):
    """Embeddable camera preview display."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._latest_frame: np.ndarray | None = None
        self._frame_counter: int = 0
        self._latest_frame_time: float = 0.0
        self._zoom_percent: float | None = None
        self._view_center = np.array([0.5, 0.5], dtype=np.float64)
        self._pan_start_pos: QPointF | None = None
        self._pan_start_center: np.ndarray | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = _PreviewLabel(self)
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
        if self._zoom_percent is not None:
            self._render_zoomed_frame(frame)
            return

        frame = self._frame_for_display(frame)
        pixmap = self._pixmap_from_frame(frame)
        self._label.setPixmap(
            pixmap.scaled(
                self._label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def _pixmap_from_frame(self, frame: np.ndarray) -> QPixmap:
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

        return QPixmap.fromImage(qimg)

    def _render_zoomed_frame(self, frame: np.ndarray) -> None:
        label_size = self._label.size()
        view_w = max(1, int(label_size.width()))
        view_h = max(1, int(label_size.height()))
        bounds = self._zoom_crop_bounds(frame, self._zoom_percent or 100.0)
        if bounds is None:
            pixmap = self._pixmap_from_frame(self._frame_for_display(frame))
            self._label.setPixmap(
                pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            return

        x0, y0, x1, y1 = bounds
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return
        if cv2 is not None:
            interpolation = (
                cv2.INTER_NEAREST
                if (self._zoom_percent or 100.0) >= 100.0 else cv2.INTER_AREA
            )
            shown = cv2.resize(crop, (view_w, view_h), interpolation=interpolation)
        else:
            y_idx = np.linspace(0, crop.shape[0] - 1, view_h).astype(np.intp)
            x_idx = np.linspace(0, crop.shape[1] - 1, view_w).astype(np.intp)
            shown = crop[np.ix_(y_idx, x_idx)]
        self._label.setPixmap(self._pixmap_from_frame(np.ascontiguousarray(shown)))

    def set_placeholder_text(self, text: str) -> None:
        """Show placeholder text when no camera is active."""
        self._label.clear()
        self._label.setText(text)

    def get_latest_frame(self) -> np.ndarray | None:
        """Return the most recently displayed frame."""
        return self._latest_frame

    def fit_to_window(self) -> None:
        self._zoom_percent = None
        self._pan_start_pos = None
        self._pan_start_center = None
        if self._latest_frame is not None:
            self._render_frame(self._latest_frame)

    def set_zoom_percent(self, percent: float, anchor_pos: QPointF | None = None) -> None:
        if self._latest_frame is None:
            self._zoom_percent = float(np.clip(percent, 10.0, 2000.0))
            return
        old_point = (
            self._image_point_at_label_position(anchor_pos)
            if anchor_pos is not None else None
        )
        self._zoom_percent = float(np.clip(percent, 10.0, 2000.0))
        if old_point is not None:
            self._set_center_for_anchor(old_point, anchor_pos)
        self._render_frame(self._latest_frame)

    def zoom_in(self) -> None:
        current = self._zoom_percent or max(100.0, self._fit_zoom_percent())
        self.set_zoom_percent(current * 1.25)

    def zoom_out(self) -> None:
        current = self._zoom_percent or max(100.0, self._fit_zoom_percent())
        self.set_zoom_percent(current / 1.25)

    @property
    def zoom_percent(self) -> float | None:
        return self._zoom_percent

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

    def _fit_zoom_percent(self) -> float:
        if self._latest_frame is None:
            return 100.0
        h, w = self._latest_frame.shape[:2]
        size = self._label.size()
        scale = min(size.width() / max(w, 1), size.height() / max(h, 1))
        return max(1.0, scale * 100.0)

    def _zoom_crop_bounds(
        self,
        frame: np.ndarray,
        zoom_percent: float,
    ) -> tuple[int, int, int, int] | None:
        h, w = frame.shape[:2]
        size = self._label.size()
        view_w = max(1, int(size.width()))
        view_h = max(1, int(size.height()))
        factor = max(0.001, zoom_percent / 100.0)
        fit_scale = min(view_w / max(w, 1), view_h / max(h, 1))
        if factor <= fit_scale:
            return None

        aspect = view_w / max(view_h, 1)
        crop_w = min(view_w / factor, float(w))
        crop_h = crop_w / aspect
        if crop_h > h:
            crop_h = float(h)
            crop_w = crop_h * aspect
        crop_w = min(crop_w, float(w))
        crop_h = min(crop_h, float(h))

        cx = float(self._view_center[0]) * max(w - 1, 1)
        cy = float(self._view_center[1]) * max(h - 1, 1)
        x0 = min(max(0.0, cx - crop_w * 0.5), max(0.0, w - crop_w))
        y0 = min(max(0.0, cy - crop_h * 0.5), max(0.0, h - crop_h))
        x1 = min(float(w), x0 + crop_w)
        y1 = min(float(h), y0 + crop_h)
        ix0 = int(np.floor(x0))
        iy0 = int(np.floor(y0))
        ix1 = min(int(np.ceil(x1)), w)
        iy1 = min(int(np.ceil(y1)), h)
        return ix0, iy0, max(ix1, ix0 + 1), max(iy1, iy0 + 1)

    def _image_point_at_label_position(self, pos: QPointF | None) -> np.ndarray | None:
        if pos is None or self._latest_frame is None:
            return None
        h, w = self._latest_frame.shape[:2]
        if self._zoom_percent is None:
            size = self._label.size()
            scale = min(size.width() / max(w, 1), size.height() / max(h, 1))
            shown_w = w * scale
            shown_h = h * scale
            left = (size.width() - shown_w) * 0.5
            top = (size.height() - shown_h) * 0.5
            x = (float(pos.x()) - left) / max(scale, 1e-9)
            y = (float(pos.y()) - top) / max(scale, 1e-9)
            return np.array([np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)])

        bounds = self._zoom_crop_bounds(self._latest_frame, self._zoom_percent)
        if bounds is None:
            old_zoom = self._zoom_percent
            self._zoom_percent = None
            point = self._image_point_at_label_position(pos)
            self._zoom_percent = old_zoom
            return point
        x0, y0, x1, y1 = bounds
        x = x0 + float(pos.x()) / max(self._label.width(), 1) * (x1 - x0)
        y = y0 + float(pos.y()) / max(self._label.height(), 1) * (y1 - y0)
        return np.array([np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)])

    def _set_center_for_anchor(self, image_point: np.ndarray, pos: QPointF | None) -> None:
        if pos is None or self._latest_frame is None:
            return
        h, w = self._latest_frame.shape[:2]
        bounds = self._zoom_crop_bounds(self._latest_frame, self._zoom_percent or 100.0)
        if bounds is None:
            self._view_center = np.array([0.5, 0.5], dtype=np.float64)
            return
        x0, y0, x1, y1 = bounds
        crop_w = x1 - x0
        crop_h = y1 - y0
        desired_x0 = image_point[0] - float(pos.x()) / max(self._label.width(), 1) * crop_w
        desired_y0 = image_point[1] - float(pos.y()) / max(self._label.height(), 1) * crop_h
        cx = desired_x0 + crop_w * 0.5
        cy = desired_y0 + crop_h * 0.5
        self._view_center = np.array([
            np.clip(cx / max(w - 1, 1), 0.0, 1.0),
            np.clip(cy / max(h - 1, 1), 0.0, 1.0),
        ], dtype=np.float64)

    def _handle_wheel_zoom(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        current = self._zoom_percent or max(100.0, self._fit_zoom_percent())
        factor = 1.25 if delta > 0 else 0.8
        self.set_zoom_percent(current * factor, event.position())
        event.accept()

    def _start_pan(self, pos: QPointF) -> None:
        if self._zoom_percent is None:
            return
        self._pan_start_pos = QPointF(pos)
        self._pan_start_center = self._view_center.copy()
        self._label.setCursor(Qt.ClosedHandCursor)

    def _update_pan(self, pos: QPointF) -> bool:
        if (
            self._zoom_percent is None
            or self._pan_start_pos is None
            or self._pan_start_center is None
            or self._latest_frame is None
        ):
            return False
        h, w = self._latest_frame.shape[:2]
        factor = max(0.001, self._zoom_percent / 100.0)
        dx_img = (float(pos.x()) - float(self._pan_start_pos.x())) / factor
        dy_img = (float(pos.y()) - float(self._pan_start_pos.y())) / factor
        self._view_center = np.array([
            np.clip(self._pan_start_center[0] - dx_img / max(w - 1, 1), 0.0, 1.0),
            np.clip(self._pan_start_center[1] - dy_img / max(h - 1, 1), 0.0, 1.0),
        ], dtype=np.float64)
        self._render_frame(self._latest_frame)
        return True

    def _end_pan(self) -> None:
        self._pan_start_pos = None
        self._pan_start_center = None
        self._label.unsetCursor()

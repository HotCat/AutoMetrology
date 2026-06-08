"""Visual image ROI selector for registration fiducials."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QImage, QPainter, QPen, QColor, QPixmap, QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDialogButtonBox, QWidget,
)


class ROIImageWidget(QWidget):
    """Image display widget with two draggable ROI rectangles."""

    COLORS = [QColor(255, 190, 40), QColor(60, 220, 255)]

    def __init__(self, image: np.ndarray, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(900, 620)
        self.setMouseTracking(True)
        self._image = image
        self._pixmap = self._to_pixmap(image)
        h, w = image.shape[:2]
        self._image_w = w
        self._image_h = h
        self._image_rect = QRectF()
        self._rois: list[Optional[QRectF]] = [None, None]
        self._active = 0
        self._drawing = False
        self._start_img = QPointF()
        self._status = "Select P1 or P2, then drag a box around the fiducial."

    def set_active(self, index: int) -> None:
        self._active = 0 if index <= 0 else 1
        self.update()

    def active(self) -> int:
        return self._active

    def set_rois(self, rois: list[Optional[tuple[int, int, int, int]]]) -> None:
        for i, roi in enumerate(rois[:2]):
            if roi is None:
                self._rois[i] = None
                continue
            x, y, w, h = roi
            self._rois[i] = QRectF(float(x), float(y), float(w), float(h)).normalized()
        self.update()

    def rois(self) -> list[Optional[tuple[int, int, int, int]]]:
        out: list[Optional[tuple[int, int, int, int]]] = []
        for rect in self._rois:
            if rect is None or rect.width() < 1 or rect.height() < 1:
                out.append(None)
                continue
            x = max(0, min(self._image_w - 1, int(round(rect.x()))))
            y = max(0, min(self._image_h - 1, int(round(rect.y()))))
            w = max(1, min(self._image_w - x, int(round(rect.width()))))
            h = max(1, min(self._image_h - y, int(round(rect.height()))))
            out.append((x, y, w, h))
        return out

    def clear_active(self) -> None:
        self._rois[self._active] = None
        self.update()

    def status_text(self) -> str:
        rois = self.rois()
        parts = []
        for i, roi in enumerate(rois):
            if roi is None:
                parts.append(f"P{i + 1}: unset")
            else:
                parts.append(f"P{i + 1}: {roi[0]},{roi[1]},{roi[2]},{roi[3]}")
        return "  |  ".join(parts)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), QColor(17, 17, 17))
            self._draw_image(painter)
            self._draw_rois(painter)
            self._draw_status(painter)
        finally:
            painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        img_pt = self._widget_to_image(event.position())
        if img_pt is None:
            return
        self._drawing = True
        self._start_img = img_pt
        self._rois[self._active] = QRectF(img_pt, img_pt)
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if not self._drawing:
            return
        img_pt = self._widget_to_image(event.position())
        if img_pt is None:
            img_pt = self._clamp_image_point(event.position())
        self._rois[self._active] = QRectF(self._start_img, img_pt).normalized()
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.LeftButton or not self._drawing:
            return
        self._drawing = False
        rect = self._rois[self._active]
        if rect is not None and (rect.width() < 5 or rect.height() < 5):
            self._rois[self._active] = None
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.update()

    def _draw_image(self, painter: QPainter) -> None:
        if self._pixmap.isNull():
            return
        scale = min(self.width() / self._image_w, self.height() / self._image_h)
        draw_w = self._image_w * scale
        draw_h = self._image_h * scale
        x = (self.width() - draw_w) / 2.0
        y = (self.height() - draw_h) / 2.0
        self._image_rect = QRectF(x, y, draw_w, draw_h)
        painter.drawPixmap(self._image_rect.toRect(), self._pixmap)

    def _draw_rois(self, painter: QPainter) -> None:
        font = QFont("Arial", 12, QFont.Bold)
        painter.setFont(font)
        for i, rect in enumerate(self._rois):
            if rect is None:
                continue
            color = QColor(self.COLORS[i])
            color.setAlpha(230 if i == self._active else 170)
            pen = QPen(color, 3 if i == self._active else 2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            wr = self._image_rect_to_widget(rect)
            painter.drawRect(wr)
            painter.drawText(wr.topLeft() + QPointF(6, -6), f"P{i + 1}")

    def _draw_status(self, painter: QPainter) -> None:
        painter.setPen(QColor(220, 220, 220))
        painter.fillRect(0, self.height() - 28, self.width(), 28, QColor(0, 0, 0, 160))
        painter.drawText(10, self.height() - 10, self.status_text())

    def _widget_to_image(self, pos: QPointF) -> Optional[QPointF]:
        if not self._image_rect.contains(pos):
            return None
        sx = self._image_w / self._image_rect.width()
        sy = self._image_h / self._image_rect.height()
        x = (pos.x() - self._image_rect.x()) * sx
        y = (pos.y() - self._image_rect.y()) * sy
        return QPointF(
            max(0.0, min(float(self._image_w - 1), x)),
            max(0.0, min(float(self._image_h - 1), y)),
        )

    def _clamp_image_point(self, pos: QPointF) -> QPointF:
        x = max(self._image_rect.left(), min(self._image_rect.right(), pos.x()))
        y = max(self._image_rect.top(), min(self._image_rect.bottom(), pos.y()))
        mapped = self._widget_to_image(QPointF(x, y))
        return mapped if mapped is not None else QPointF()

    def _image_rect_to_widget(self, rect: QRectF) -> QRectF:
        sx = self._image_rect.width() / self._image_w
        sy = self._image_rect.height() / self._image_h
        return QRectF(
            self._image_rect.x() + rect.x() * sx,
            self._image_rect.y() + rect.y() * sy,
            rect.width() * sx,
            rect.height() * sy,
        )

    @staticmethod
    def _to_pixmap(image: np.ndarray) -> QPixmap:
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            gray = image if image.ndim == 2 else image[:, :, 0]
            h, w = gray.shape
            qimg = QImage(gray.data, w, h, w, QImage.Format_Grayscale8).copy()
        else:
            rgb = image[:, :, ::-1].copy()
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)


class ROISelectorDialog(QDialog):
    """Dialog for visually selecting two fiducial search ROIs."""

    def __init__(
        self,
        image: np.ndarray,
        rois: list[Optional[tuple[int, int, int, int]]] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Fiducial Search ROIs")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        self.resize(1180, 820)
        self.setStyleSheet("""
            QDialog { background: #1e1e1e; color: #cccccc; }
            QLabel { color: #cccccc; }
            QPushButton {
                background: #333; color: #ccc; border: 1px solid #555;
                padding: 6px 12px; border-radius: 3px;
            }
            QPushButton:hover { background: #444; }
            QPushButton:checked { background: #264f78; color: white; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        toolbar = QHBoxLayout()
        self._btn_p1 = QPushButton("Draw ROI P1")
        self._btn_p1.setCheckable(True)
        self._btn_p2 = QPushButton("Draw ROI P2")
        self._btn_p2.setCheckable(True)
        self._btn_p1.setChecked(True)
        self._btn_clear = QPushButton("Clear Active")
        toolbar.addWidget(self._btn_p1)
        toolbar.addWidget(self._btn_p2)
        toolbar.addWidget(self._btn_clear)
        toolbar.addStretch()
        hint = QLabel("Drag on the image to draw the active ROI. Existing boxes show saved search areas.")
        hint.setStyleSheet("color: #999; font-size: 11px;")
        toolbar.addWidget(hint)
        layout.addLayout(toolbar)

        self._image_widget = ROIImageWidget(image)
        if rois:
            self._image_widget.set_rois(rois)
        layout.addWidget(self._image_widget, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._btn_p1.clicked.connect(lambda: self._set_active(0))
        self._btn_p2.clicked.connect(lambda: self._set_active(1))
        self._btn_clear.clicked.connect(self._image_widget.clear_active)

    def _set_active(self, index: int) -> None:
        self._image_widget.set_active(index)
        self._btn_p1.setChecked(index == 0)
        self._btn_p2.setChecked(index == 1)

    def get_rois(self) -> list[Optional[tuple[int, int, int, int]]]:
        return self._image_widget.rois()

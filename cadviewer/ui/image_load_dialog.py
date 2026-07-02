"""
ImageLoadDialog — dialog for loading a telecentric image with pixel size.

Supports two sources:
  1. File: load from disk (PNG, BMP, TIF)
  2. Camera: capture live frame, apply lens undistortion if calibrated
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QDoubleSpinBox, QPushButton,
    QFileDialog, QGroupBox, QDialogButtonBox, QRadioButton,
    QButtonGroup, QCheckBox,
)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def _undistort(frame: np.ndarray, config) -> tuple[np.ndarray, bool]:
    """Apply lens undistortion if calibration data is available."""
    if not HAS_CV2 or config is None:
        return frame, False
    lc = config.lens_calibration
    if not lc.calibrated:
        return frame, False
    mtx = lc.get_camera_matrix()
    dist = lc.get_dist_coeffs()
    if mtx is None or dist is None:
        return frame, False
    return cv2.undistort(frame, mtx, dist), True


def _frame_to_pixmap(frame: np.ndarray, max_h: int = 200) -> QPixmap:
    """Convert BGR or grayscale numpy array to a scaled QPixmap."""
    if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
        if frame.ndim == 3:
            frame = frame[:, :, 0]
        h, w = frame.shape
        qimg = QImage(frame.data, w, h, w, QImage.Format_Grayscale8).copy()
    else:
        rgb = frame[:, :, ::-1].copy()
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    pm = QPixmap.fromImage(qimg)
    return pm.scaledToHeight(max_h, Qt.SmoothTransformation)


class ImageLoadDialog(QDialog):
    """Dialog for loading a telecentric product image."""

    def __init__(
        self,
        parent=None,
        default_pixel_size: float = 0.01,
        camera=None,
        config=None,
    ) -> None:
        super().__init__(parent)
        self._camera = camera
        self._config = config
        self._captured_frame: Optional[np.ndarray] = None
        self._captured_calibration_applied: bool = False
        self._latest_cam_frame: Optional[np.ndarray] = None

        self.setWindowTitle("Load Telecentric Image")
        self.setMinimumWidth(480)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; color: #cccccc; }
            QLabel { color: #cccccc; }
            QLineEdit, QDoubleSpinBox {
                background: #2d2d2d; color: #cccccc;
                border: 1px solid #3d3d3d; border-radius: 3px; padding: 4px;
            }
            QPushButton {
                background: #333; color: #ccc; border: 1px solid #555;
                padding: 6px 16px; border-radius: 3px;
            }
            QPushButton:hover { background: #444; }
            QPushButton:disabled { background: #222; color: #666; }
            QGroupBox {
                color: #aaa; border: 1px solid #333;
                border-radius: 4px; margin-top: 8px; padding-top: 14px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; }
            QRadioButton { color: #ccc; spacing: 6px; }
        """)

        layout = QVBoxLayout(self)

        # ── Source selection ──────────────────────────────────────────
        src_group = QGroupBox("Image Source")
        src_layout = QVBoxLayout(src_group)

        radio_row = QHBoxLayout()
        self._src_buttons = QButtonGroup(self)
        self._radio_file = QRadioButton("From File")
        self._radio_cam = QRadioButton("From Camera")
        self._radio_file.setChecked(True)
        self._src_buttons.addButton(self._radio_file)
        self._src_buttons.addButton(self._radio_cam)
        if camera is None:
            self._radio_cam.setEnabled(False)
        radio_row.addWidget(self._radio_file)
        radio_row.addWidget(self._radio_cam)
        radio_row.addStretch()
        src_layout.addLayout(radio_row)

        # File picker row
        self._file_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Select PNG, BMP, or TIF file...")
        self._path_edit.setReadOnly(True)
        self._file_row.addWidget(self._path_edit, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)
        self._file_row.addWidget(browse_btn)
        src_layout.addLayout(self._file_row)

        # Camera capture row
        self._cam_row = QHBoxLayout()
        self._cam_preview = QLabel("Camera preview")
        self._cam_preview.setAlignment(Qt.AlignCenter)
        self._cam_preview.setMinimumHeight(180)
        self._cam_preview.setStyleSheet(
            "background: #111; border: 1px solid #333; color: #555;"
        )
        self._cam_row.addWidget(self._cam_preview, 1)

        cam_btn_col = QVBoxLayout()
        self._btn_capture = QPushButton("Capture")
        self._btn_capture.clicked.connect(self._capture_frame)
        cam_btn_col.addWidget(self._btn_capture)
        cam_btn_col.addStretch()
        self._cam_row.addLayout(cam_btn_col)
        src_layout.addLayout(self._cam_row)
        self._cam_row_widget = self._cam_row.itemAt(0).widget().parent()

        self._skip_calibration = QCheckBox("Ignore saved lens calibration for this image")
        self._skip_calibration.setToolTip(
            "Load or capture this image without applying the saved camera/lens calibration"
        )
        self._skip_calibration.toggled.connect(self._on_skip_calibration_changed)
        src_layout.addWidget(self._skip_calibration)

        layout.addWidget(src_group)

        # ── Preview ───────────────────────────────────────────────────
        self._preview = QLabel("No image selected")
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setMinimumHeight(120)
        self._preview.setStyleSheet(
            "background: #111; border: 1px solid #333; color: #555;"
        )
        layout.addWidget(self._preview)

        # ── Pixel size ────────────────────────────────────────────────
        ps_group = QGroupBox("Pixel Size")
        ps_layout = QFormLayout(ps_group)
        self._pixel_size = QDoubleSpinBox()
        self._pixel_size.setRange(0.0001, 100.0)
        self._pixel_size.setValue(default_pixel_size)
        self._pixel_size.setDecimals(4)
        self._pixel_size.setSuffix(" mm/pixel")
        ps_layout.addRow("Pixel Size:", self._pixel_size)
        layout.addWidget(ps_group)

        # ── Buttons ───────────────────────────────────────────────────
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        # Toggle visibility based on source
        self._radio_cam.toggled.connect(self._on_source_changed)
        self._on_source_changed(False)

        # Connect camera if available
        if self._camera is not None:
            self._camera.signals.frame_ready.connect(self._show_cam_frame)

    def _on_source_changed(self, camera_selected: bool) -> None:
        self._path_edit.setVisible(not camera_selected)
        # Hide/show file browse button
        for i in range(self._file_row.count()):
            w = self._file_row.itemAt(i).widget()
            if w is not None:
                w.setVisible(not camera_selected)
        self._cam_preview.setVisible(camera_selected)
        self._btn_capture.setVisible(camera_selected)

    def _show_cam_frame(self, frame: np.ndarray) -> None:
        """Store and show live camera frame in preview."""
        self._latest_cam_frame = frame
        if self._radio_cam.isChecked():
            self._cam_preview.setPixmap(_frame_to_pixmap(frame, 180))

    def _capture_frame(self) -> None:
        """Capture latest camera frame, apply undistortion."""
        if self._latest_cam_frame is None:
            return
        frame = self._latest_cam_frame.copy()
        # Ensure BGR
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # Apply lens undistortion unless this image is from an uncalibrated test camera.
        if self.skip_calibration():
            corrected, applied = frame, False
        else:
            corrected, applied = _undistort(frame, self._config)
        self._captured_frame = corrected
        self._captured_calibration_applied = applied
        self._preview.setPixmap(_frame_to_pixmap(corrected, 200))

    def _on_skip_calibration_changed(self) -> None:
        """Rebuild captured camera preview if the temporary calibration policy changes."""
        if self._radio_cam.isChecked() and self._captured_frame is not None:
            self._capture_frame()

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", str(Path.cwd()),
            "Images (*.png *.bmp *.tif *.tiff);;All Files (*)",
        )
        if path:
            self._path_edit.setText(path)
            self._captured_frame = None
            self._captured_calibration_applied = False
            img = cv2.imread(path)
            if img is not None:
                self._preview.setPixmap(_frame_to_pixmap(img, 200))

    def get_values(self) -> Tuple[str, float]:
        return self._path_edit.text(), self._pixel_size.value()

    def get_captured_frame(self) -> Optional[np.ndarray]:
        """Return the camera-captured frame, or None."""
        return self._captured_frame

    def calibration_applied(self) -> bool:
        return bool(self._captured_calibration_applied)

    def skip_calibration(self) -> bool:
        return bool(self._skip_calibration.isChecked())

    def closeEvent(self, event) -> None:
        if self._camera is not None:
            try:
                self._camera.signals.frame_ready.disconnect(self._show_cam_frame)
            except (RuntimeError, TypeError):
                pass
        super().closeEvent(event)

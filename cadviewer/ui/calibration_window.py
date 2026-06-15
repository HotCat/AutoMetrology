"""
CalibrationWindow — dedicated window for camera and pixel-size calibration.

Two tabs:
  1. Pixel Size Calibration: load a chessboard photo, detect corners, compute mm/px
  2. Lens Calibration: capture multiple chessboard images, run cv2.calibrateCamera

Shared chessboard parameters (cols, rows, cell size) sit above the tabs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QIcon
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QDoubleSpinBox, QPushButton,
    QFileDialog, QGroupBox, QTabWidget, QWidget,
    QSpinBox, QRadioButton, QButtonGroup, QListWidget,
    QListWidgetItem, QSplitter, QAbstractItemView,
)

from ..core.config import AppConfig
from ..camera.preview_widget import CameraPreviewWidget

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ── Stylesheet ──────────────────────────────────────────────────────────

_DARK_STYLE = """
    QDialog { background-color: #1e1e1e; color: #cccccc; }
    QLabel { color: #cccccc; }
    QLineEdit, QDoubleSpinBox, QSpinBox {
        background: #2d2d2d; color: #cccccc;
        border: 1px solid #3d3d3d; border-radius: 3px; padding: 4px;
    }
    QPushButton {
        background: #333; color: #ccc; border: 1px solid #555;
        padding: 6px 14px; border-radius: 3px;
    }
    QPushButton:hover { background: #444; }
    QPushButton:disabled { background: #222; color: #666; }
    QGroupBox {
        color: #aaa; border: 1px solid #333;
        border-radius: 4px; margin-top: 8px; padding-top: 14px;
    }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; }
    QTabWidget::pane { border: 1px solid #333; background: #1e1e1e; }
    QTabBar::tab {
        background: #2d2d2d; color: #aaa; padding: 8px 20px;
        border: 1px solid #333; border-bottom: none; border-radius: 4px 4px 0 0;
    }
    QTabBar::tab:selected { background: #1e1e1e; color: #ddd; }
    QTabBar::tab:hover { background: #333; }
    QRadioButton { color: #ccc; spacing: 6px; }
    QListWidget {
        background: #1a1a1a; border: 1px solid #333;
        color: #ccc;
    }
    QListWidget::item { padding: 2px; }
    QListWidget::item:selected { background: #264f78; }
"""


# ── Helpers ─────────────────────────────────────────────────────────────



def _to_gray_image(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0]
    return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

def _numpy_to_pixmap(arr: np.ndarray, max_size: int = 400) -> QPixmap:
    """Convert numpy array (BGR or grayscale) to QPixmap, scaled to fit max_size."""
    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    else:
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = min(max_size / w, max_size / h, 1.0)
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
    qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                  rgb.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def _thumbnail_with_badge(pixmap: QPixmap, ok: bool) -> QPixmap:
    """Draw a green-check or red-X badge on the bottom-right of a thumbnail."""
    size = 120
    scaled = pixmap.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    result = QPixmap(size, size)
    result.fill(QColor(30, 30, 30))
    x = (size - scaled.width()) // 2
    y = (size - scaled.height()) // 2
    painter = QPainter(result)
    painter.drawPixmap(x, y, scaled)
    # Badge
    painter.setFont(QFont("Sans", 16, QFont.Bold))
    if ok:
        painter.setPen(QColor("#66bb6a"))
        painter.drawText(size - 22, size - 6, "✓")
    else:
        painter.setPen(QColor("#ef5350"))
        painter.drawText(size - 22, size - 6, "✗")
    painter.end()
    return result


# ── Collected image data ────────────────────────────────────────────────

@dataclass
class _CollectedImage:
    image: np.ndarray
    corners: Optional[np.ndarray]
    detected: bool
    source: str


# ── Pixel Size Calibration Tab ──────────────────────────────────────────

class _PixelSizeTab(QWidget):
    """Tab for computing mm/pixel from a single chessboard photo."""

    def __init__(self, parent: CalibrationWindow) -> None:
        super().__init__(parent)
        self._win = parent
        self._computed_pixel_size: Optional[float] = None
        self._captured_frame: Optional[np.ndarray] = None
        self._latest_cam_frame: Optional[np.ndarray] = None

        layout = QVBoxLayout(self)

        # ── Source selection ─────────────────────────────────────────
        src_group = QGroupBox("Image Source")
        src_layout = QVBoxLayout(src_group)

        radio_row = QHBoxLayout()
        self._src_group = QButtonGroup(self)
        self._radio_file = QRadioButton("From File")
        self._radio_cam = QRadioButton("From Camera")
        self._radio_file.setChecked(True)
        self._src_group.addButton(self._radio_file)
        self._src_group.addButton(self._radio_cam)
        if parent._camera is None:
            self._radio_cam.setEnabled(False)
        radio_row.addWidget(self._radio_file)
        radio_row.addWidget(self._radio_cam)
        radio_row.addStretch()
        src_layout.addLayout(radio_row)

        # File picker row
        self._file_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Photo of printed chessboard pattern...")
        self._path_edit.setReadOnly(True)
        self._file_row.addWidget(self._path_edit, 1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        self._file_row.addWidget(browse)
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
        cap_col = QVBoxLayout()
        self._btn_capture = QPushButton("Capture")
        self._btn_capture.clicked.connect(self._capture_frame)
        cap_col.addWidget(self._btn_capture)
        cap_col.addStretch()
        self._cam_row.addLayout(cap_col)
        src_layout.addLayout(self._cam_row)

        layout.addWidget(src_group)

        # ── Preview ───────────────────────────────────────────────────
        self._preview = QLabel("No image loaded")
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setMinimumHeight(200)
        self._preview.setStyleSheet(
            "background: #111; border: 1px solid #333; color: #555;"
        )
        layout.addWidget(self._preview)

        # ── Calibrate ─────────────────────────────────────────────────
        cal_row = QHBoxLayout()
        self._btn_calibrate = QPushButton("Calibrate Pixel Size")
        self._btn_calibrate.clicked.connect(self._calibrate)
        cal_row.addWidget(self._btn_calibrate)
        cal_row.addStretch()
        layout.addLayout(cal_row)

        self._result = QLabel("")
        self._result.setWordWrap(True)
        layout.addWidget(self._result)

        layout.addStretch()

        # Connect source toggle
        self._radio_cam.toggled.connect(self._on_source_changed)
        self._on_source_changed(False)

        # Connect camera if available
        if parent._camera is not None:
            parent._camera.signals.frame_ready.connect(self._show_cam_frame)

    def _on_source_changed(self, camera_selected: bool) -> None:
        for i in range(self._file_row.count()):
            w = self._file_row.itemAt(i).widget()
            if w is not None:
                w.setVisible(not camera_selected)
        self._cam_preview.setVisible(camera_selected)
        self._btn_capture.setVisible(camera_selected)

    def _show_cam_frame(self, frame: np.ndarray) -> None:
        self._latest_cam_frame = frame
        if self._radio_cam.isChecked():
            self._cam_preview.setPixmap(_numpy_to_pixmap(frame, 180))

    def _capture_frame(self) -> None:
        """Capture latest camera frame, apply undistortion."""
        if self._latest_cam_frame is None:
            self._result.setText("No frame available.")
            self._result.setStyleSheet("color: #ef5350;")
            return
        frame = self._latest_cam_frame.copy()
        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            frame = cv2.cvtColor(frame if frame.ndim == 2 else frame[:, :, 0],
                                 cv2.COLOR_GRAY2BGR)
        corrected = self._undistort(frame)
        self._captured_frame = corrected
        self._preview.setPixmap(_numpy_to_pixmap(corrected, 400))

    def _undistort(self, frame: np.ndarray) -> np.ndarray:
        """Apply lens undistortion if calibration data exists."""
        lc = self._win._config.lens_calibration
        if not lc.calibrated:
            return frame
        mtx = lc.get_camera_matrix()
        dist = lc.get_dist_coeffs()
        if mtx is None or dist is None:
            return frame
        return cv2.undistort(frame, mtx, dist)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Chessboard Image", str(Path.cwd()),
            "Images (*.png *.bmp *.tif *.tiff);;All Files (*)",
        )
        if path:
            self._path_edit.setText(path)
            self._computed_pixel_size = None
            self._captured_frame = None
            img = cv2.imread(path)
            if img is not None:
                self._preview.setPixmap(_numpy_to_pixmap(img, 400))

    def _calibrate(self) -> None:
        if not HAS_CV2:
            self._result.setText("Error: OpenCV not available")
            self._result.setStyleSheet("color: #ef5350;")
            return

        # Get image: either from file or from captured frame
        if self._radio_cam.isChecked() and self._captured_frame is not None:
            img = self._captured_frame
        else:
            path = self._path_edit.text().strip()
            if not path or not Path(path).exists():
                self._result.setText("Select a chessboard image first.")
                self._result.setStyleSheet("color: #ef5350;")
                return
            img = cv2.imread(path)
            if img is None:
                self._result.setText("Cannot read image file.")
                self._result.setStyleSheet("color: #ef5350;")
                return

        cols = self._win._cb_col.value()
        rows = self._win._cb_row.value()
        cell_mm = self._win._cb_cell.value()

        if img.ndim == 2:
            gray = img
        elif img.shape[2] == 1:
            gray = img[:, :, 0]
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if not found:
            self._result.setText(
                f"Chessboard ({cols}×{rows}) not detected in image."
            )
            self._result.setStyleSheet("color: #ef5350;")
            return

        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )

        pts = corners.reshape(-1, 2)
        h_dists, v_dists = [], []
        for r in range(rows):
            for c in range(cols - 1):
                d = np.linalg.norm(pts[r * cols + c + 1] - pts[r * cols + c])
                h_dists.append(d)
        for r in range(rows - 1):
            for c in range(cols):
                d = np.linalg.norm(pts[(r + 1) * cols + c] - pts[r * cols + c])
                v_dists.append(d)

        avg_px = (np.mean(h_dists) + np.mean(v_dists)) / 2.0
        pixel_size = float(cell_mm / avg_px)
        self._computed_pixel_size = pixel_size
        self._win._config.pixel_size_mm = pixel_size
        self._win._config.save()

        # Draw corners on preview
        vis = img.copy()
        cv2.drawChessboardCorners(vis, (cols, rows), corners, True)
        self._preview.setPixmap(_numpy_to_pixmap(vis, 400))

        self._result.setText(
            f"Detected {cols}×{rows} — {avg_px:.2f} px/cell → "
            f"{pixel_size:.4f} mm/px"
        )
        self._result.setStyleSheet("color: #66bb6a; font-weight: bold;")

    def get_pixel_size(self) -> Optional[float]:
        return self._computed_pixel_size

    def cleanup(self) -> None:
        if self._win._camera is not None:
            try:
                self._win._camera.signals.frame_ready.disconnect(self._show_cam_frame)
            except (RuntimeError, TypeError):
                pass


# ── Lens Calibration Tab ────────────────────────────────────────────────

class _LensCalTab(QWidget):
    """Tab for camera lens calibration using multiple chessboard images."""

    def __init__(self, parent: CalibrationWindow, camera) -> None:
        super().__init__(parent)
        self._win = parent
        self._camera = camera
        self._collected: list[_CollectedImage] = []
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs: Optional[np.ndarray] = None
        self._rms_error: float = 0.0
        self._cal_result = None

        layout = QVBoxLayout(self)

        # ── Source selection ─────────────────────────────────────────
        src_group = QGroupBox("Image Source")
        src_layout = QVBoxLayout(src_group)

        radio_row = QHBoxLayout()
        self._src_group = QButtonGroup(self)
        self._radio_cam = QRadioButton("From Camera")
        self._radio_files = QRadioButton("From Files")
        self._radio_files.setChecked(True)
        self._src_group.addButton(self._radio_cam)
        self._src_group.addButton(self._radio_files)
        if camera is None:
            self._radio_cam.setEnabled(False)
        else:
            self._radio_cam.setChecked(True)
        radio_row.addWidget(self._radio_cam)
        radio_row.addWidget(self._radio_files)
        radio_row.addStretch()
        src_layout.addLayout(radio_row)

        # Preview area
        self._preview = CameraPreviewWidget()
        self._preview.setMinimumHeight(180)
        self._preview.setMaximumHeight(260)
        self._preview.set_placeholder_text(
            "No camera" if camera is None else "Camera not streaming"
        )
        src_layout.addWidget(self._preview)

        # Action buttons
        btn_row = QHBoxLayout()
        self._btn_capture = QPushButton("Capture Frame")
        self._btn_capture.clicked.connect(self._capture_frame)
        self._btn_capture.setEnabled(camera is not None)
        btn_row.addWidget(self._btn_capture)

        self._btn_add_files = QPushButton("Add Files...")
        self._btn_add_files.clicked.connect(self._add_files)
        btn_row.addWidget(self._btn_add_files)

        self._btn_clear = QPushButton("Clear All")
        self._btn_clear.clicked.connect(self._clear_all)
        btn_row.addWidget(self._btn_clear)
        src_layout.addLayout(btn_row)

        layout.addWidget(src_group)

        # ── Collected images grid ────────────────────────────────────
        grid_group = QGroupBox("Collected Images")
        grid_layout = QVBoxLayout(grid_group)

        self._image_list = QListWidget()
        self._image_list.setViewMode(QListWidget.IconMode)
        self._image_list.setIconSize(QSize(120, 120))
        self._image_list.setResizeMode(QListWidget.Adjust)
        self._image_list.setSelectionMode(QListWidget.SingleSelection)
        self._image_list.setMinimumHeight(140)
        grid_layout.addWidget(self._image_list)

        info_row = QHBoxLayout()
        self._count_label = QLabel("Images: 0 | Corners detected: 0")
        info_row.addWidget(self._count_label)
        info_row.addStretch()
        self._btn_remove = QPushButton("Remove Selected")
        self._btn_remove.clicked.connect(self._remove_selected)
        info_row.addWidget(self._btn_remove)
        grid_layout.addLayout(info_row)

        layout.addWidget(grid_group)

        # ── Calibration action ───────────────────────────────────────
        cal_row = QHBoxLayout()
        self._btn_run = QPushButton("Run Calibration")
        self._btn_run.setStyleSheet(
            "QPushButton { background: #264f78; color: white; font-weight: bold; }"
            "QPushButton:hover { background: #306898; }"
        )
        self._btn_run.clicked.connect(self._run_calibration)
        cal_row.addWidget(self._btn_run)
        cal_row.addStretch()
        layout.addLayout(cal_row)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        # ── Results ──────────────────────────────────────────────────
        res_group = QGroupBox("Results")
        res_layout = QVBoxLayout(res_group)
        self._result_text = QLabel("No calibration results yet.")
        self._result_text.setWordWrap(True)
        self._result_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        res_layout.addWidget(self._result_text)

        save_row = QHBoxLayout()
        self._btn_save = QPushButton("Save to Config")
        self._btn_save.clicked.connect(self._save_to_config)
        self._btn_save.setEnabled(False)
        save_row.addWidget(self._btn_save)
        save_row.addStretch()
        res_layout.addLayout(save_row)

        layout.addWidget(res_group)

        # Connect source toggle
        self._radio_cam.toggled.connect(self._on_source_changed)
        self._on_source_changed(self._radio_cam.isChecked())

    # ── Source toggle ────────────────────────────────────────────────

    def _on_source_changed(self, camera_selected: bool) -> None:
        if camera_selected and self._camera is not None:
            self._camera.signals.frame_ready.connect(self._preview.display_frame)
            self._btn_capture.setEnabled(True)
            self._preview.set_placeholder_text("Waiting for camera...")
        else:
            if self._camera is not None:
                try:
                    self._camera.signals.frame_ready.disconnect(self._preview.display_frame)
                except (RuntimeError, TypeError):
                    pass
            self._btn_capture.setEnabled(False)
            self._preview.set_placeholder_text("Load images from files")

    # ── Image collection ─────────────────────────────────────────────

    def _capture_frame(self) -> None:
        frame = self._preview.get_latest_frame()
        if frame is None:
            self._status_label.setText("No frame available to capture.")
            return
        self._add_image(frame, "camera")

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Chessboard Images", str(Path.cwd()),
            "Images (*.png *.bmp *.tif *.tiff);;All Files (*)",
        )
        for p in paths:
            img = cv2.imread(p)
            if img is not None:
                self._add_image(img, Path(p).name)

    def _add_image(self, image: np.ndarray, source: str) -> None:
        cols = self._win._cb_col.value()
        rows = self._win._cb_row.value()
        corners, detected = self._detect_corners(image, cols, rows)

        # Ensure BGR format for storage
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        entry = _CollectedImage(
            image=image, corners=corners, detected=detected, source=source,
        )
        self._collected.append(entry)

        # Thumbnail
        pm = _numpy_to_pixmap(image, 120)
        icon_pm = _thumbnail_with_badge(pm, detected)
        item = QListWidgetItem(QIcon(icon_pm), "")
        item.setData(Qt.UserRole, len(self._collected) - 1)
        self._image_list.addItem(item)

        self._update_count()
        self._status_label.setText(
            f"Added: {source} — {'corners found' if detected else 'corners NOT found'}"
        )

    @staticmethod
    def _detect_corners(image: np.ndarray, cols: int, rows: int):
        """Detect chessboard corners. Returns (corners, found)."""
        if not HAS_CV2:
            return None, False
        # Handle both BGR and grayscale input
        if image.ndim == 2:
            gray = image
        elif image.shape[2] == 1:
            gray = image[:, :, 0]
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            return None, False
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        return corners, True

    def _remove_selected(self) -> None:
        idx = self._image_list.currentRow()
        if 0 <= idx < len(self._collected):
            self._collected.pop(idx)
            self._image_list.takeItem(idx)
            # Re-index remaining items
            for i in range(self._image_list.count()):
                self._image_list.item(i).setData(Qt.UserRole, i)
            self._update_count()

    def _clear_all(self) -> None:
        self._collected.clear()
        self._image_list.clear()
        self._update_count()
        self._status_label.setText("All images cleared.")

    def _update_count(self) -> None:
        total = len(self._collected)
        good = sum(1 for e in self._collected if e.detected)
        self._count_label.setText(
            f"Images: {total} | Corners detected: {good}"
        )

    # ── Calibration ──────────────────────────────────────────────────

    def _run_calibration(self) -> None:
        if not HAS_CV2:
            self._status_label.setText("Error: OpenCV not available.")
            return

        good = [e for e in self._collected if e.detected]
        if len(good) < 3:
            self._status_label.setText(
                f"Need at least 3 images with detected corners (have {len(good)})."
            )
            self._status_label.setStyleSheet("color: #ef5350;")
            return

        cols = self._win._cb_col.value()
        rows = self._win._cb_row.value()
        cell_mm = self._win._cb_cell.value()

        self._status_label.setText("Running calibration...")
        self._status_label.setStyleSheet("color: #ccc;")

        from ..calibration.calibration_manager import CalibrationManager

        mgr = CalibrationManager()
        for entry in self._collected:
            mgr.add_image(entry.image, entry.source)

        result = mgr.run_calibration(cols, rows, cell_mm)

        if not result.calibrated:
            self._status_label.setText("Calibration failed.")
            self._status_label.setStyleSheet("color: #ef5350;")
            return

        self._camera_matrix = result.camera_matrix
        self._dist_coeffs = result.dist_coeffs
        self._rms_error = result.opencv_rms
        self._cal_result = result

        # Display results
        mtx = result.camera_matrix
        dist = result.dist_coeffs
        fx, fy = mtx[0, 0], mtx[1, 1]
        cx, cy = mtx[0, 2], mtx[1, 2]
        lines = [
            f"Reprojection error (RMS): {result.opencv_rms:.4f} px",
            f"Images used: {result.image_count}",
            f"Corners: {result.corner_count}",
            "",
            f"Camera Matrix:",
            f"  fx = {fx:.2f}   fy = {fy:.2f}",
            f"  cx = {cx:.2f}   cy = {cy:.2f}",
            "",
            f"Distortion ({len(dist)} coefficients):",
        ]
        labels = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
        for i, v in enumerate(dist.flatten()):
            label = labels[i] if i < len(labels) else f"d{i}"
            lines.append(f"  {label} = {v:.6f}")

        if result.report is not None:
            lines += ["", result.report.summary()]

        self._result_text.setText("\n".join(lines))
        self._btn_save.setEnabled(True)
        self._status_label.setText(
            f"Calibration complete — RMS: {result.opencv_rms:.4f} px ({result.image_count} images)"
        )
        self._status_label.setStyleSheet("color: #66bb6a; font-weight: bold;")

    def _save_to_config(self) -> None:
        try:
            self._save_to_config_impl()
        except Exception as e:
            self._status_label.setText(f"Calibration save error: {e}")
            self._status_label.setStyleSheet("color: #ef5350;")

    def _save_to_config_impl(self) -> None:
        if self._camera_matrix is None:
            return
        cfg = self._win._config
        good = sum(1 for e in self._collected if e.detected)
        cfg.lens_calibration.set_from_results(
            self._camera_matrix, self._dist_coeffs,
            self._rms_error, good,
        )

        # Build and save coordinate correction model from detected corners
        if hasattr(self, "_cal_result") and self._cal_result is not None:
            cols = self._win._cb_col.value()
            rows = self._win._cb_row.value()
            cell_mm = self._win._cb_cell.value()

            # Build correction models in undistorted-pixel space, because
            # production registration first applies OpenCV undistortion.
            good_entries = [e for e in self._collected if e.detected]
            undistorted_sets = self._undistorted_corner_sets(
                good_entries, cols, rows, self._camera_matrix, self._dist_coeffs,
            )
            if undistorted_sets:
                from ..calibration.coordinate_correction import CoordinateTransformer
                from ..calibration.residual_map import (
                    ResidualDistortionMap, is_residual_map_safe,
                )

                samples = []
                corrections = []
                from ..calibration.calibration_manager import CalibrationManager
                for corners in undistorted_sets:
                    ideal = CalibrationManager._compute_projective_ideal_grid(
                        corners, cols, rows,
                    )
                    if ideal is None:
                        continue
                    samples.append(corners)
                    corrections.append(ideal - corners)

                residual_map = None
                if samples:
                    candidate_map = ResidualDistortionMap()
                    candidate_map.build(
                        np.vstack(samples), np.vstack(corrections),
                        image_size=(good_entries[0].image.shape[1],
                                    good_entries[0].image.shape[0]),
                        smoothing=0.01,
                    )
                    if is_residual_map_safe(candidate_map):
                        residual_map = candidate_map
                        cfg.lens_calibration.residual_map = residual_map.to_dict()
                    else:
                        cfg.lens_calibration.residual_map = {}

                transformer = CoordinateTransformer()
                model_type = "homography"
                first_corners = undistorted_sets[0]
                if residual_map is not None and residual_map.is_built:
                    first_corners = residual_map.correct(first_corners)
                success = transformer.build_from_corners(
                    first_corners, cols, rows, cell_mm, model_type,
                    image_size=(good_entries[0].image.shape[1],
                                good_entries[0].image.shape[0]),
                    image_count=len(undistorted_sets),
                )
                if success:
                    cfg.lens_calibration.coordinate_correction = transformer.get_model_dict()
                    cfg.lens_calibration.correction_model_type = model_type

        cfg.save()
        self._status_label.setText("Calibration saved to configuration.")
        self._status_label.setStyleSheet("color: #66bb6a; font-weight: bold;")


    def _undistorted_corner_sets(
        self, entries: list, cols: int, rows: int, camera_matrix, dist_coeffs,
    ) -> list[np.ndarray]:
        if camera_matrix is None or dist_coeffs is None:
            return []
        sets: list[np.ndarray] = []
        for entry in entries:
            undistorted = cv2.undistort(entry.image, camera_matrix, dist_coeffs)
            gray = _to_gray_image(undistorted)
            found, corners = cv2.findChessboardCorners(
                gray, (cols, rows),
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if found:
                corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                )
                sets.append(corners.reshape(-1, 2).astype(np.float64))
                continue
            if entry.corners is None:
                continue
            undistorted_pts = cv2.undistortPoints(
                entry.corners.astype(np.float32),
                camera_matrix, dist_coeffs,
                P=camera_matrix,
            )
            sets.append(undistorted_pts.reshape(-1, 2).astype(np.float64))
        return sets

    # ── Cleanup ──────────────────────────────────────────────────────

    def cleanup(self) -> None:
        if self._camera is not None:
            try:
                self._camera.signals.frame_ready.disconnect(self._preview.display_frame)
            except (RuntimeError, TypeError):
                pass


# ── Main Window ─────────────────────────────────────────────────────────

class CalibrationWindow(QDialog):
    """Camera calibration window with pixel-size and lens calibration tabs."""

    def __init__(self, parent=None, config: AppConfig = None,
                 camera=None) -> None:
        super().__init__(parent)
        self._config = config or AppConfig()
        self._camera = camera
        self.setWindowTitle("Camera Calibration")
        self.setMinimumSize(560, 640)
        self.setStyleSheet(_DARK_STYLE)

        layout = QVBoxLayout(self)

        # ── Shared chessboard parameters ─────────────────────────────
        cb_group = QGroupBox("Chessboard Pattern")
        cb_layout = QHBoxLayout(cb_group)

        cb_layout.addWidget(QLabel("Cols:"))
        self._cb_col = QSpinBox()
        self._cb_col.setRange(3, 30)
        self._cb_col.setValue(self._config.calibration.chessboard_cols)
        cb_layout.addWidget(self._cb_col)

        cb_layout.addWidget(QLabel("Rows:"))
        self._cb_row = QSpinBox()
        self._cb_row.setRange(3, 30)
        self._cb_row.setValue(self._config.calibration.chessboard_rows)
        cb_layout.addWidget(self._cb_row)

        cb_layout.addWidget(QLabel("Cell:"))
        self._cb_cell = QDoubleSpinBox()
        self._cb_cell.setRange(0.1, 500.0)
        self._cb_cell.setDecimals(1)
        self._cb_cell.setSuffix(" mm")
        self._cb_cell.setValue(self._config.calibration.chessboard_cell_mm)
        cb_layout.addWidget(self._cb_cell)

        cb_layout.addStretch()
        layout.addWidget(cb_group)

        # ── Tabs ─────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._pixel_tab = _PixelSizeTab(self)
        self._lens_tab = _LensCalTab(self, camera)
        self._tabs.addTab(self._pixel_tab, "Pixel Size Calibration")
        self._tabs.addTab(self._lens_tab, "Lens Calibration")
        layout.addWidget(self._tabs)

        # ── Bottom buttons ───────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        # Show previous lens calibration results if available
        if self._config.lens_calibration.calibrated:
            lc = self._config.lens_calibration
            self._lens_tab._status_label.setText(
                f"Previously calibrated: RMS={lc.reprojection_error:.4f} px "
                f"({lc.image_count} images)"
            )
            self._lens_tab._status_label.setStyleSheet("color: #66bb6a;")

    def get_chessboard_params(self) -> dict:
        return {
            "cols": int(self._cb_col.value()),
            "rows": int(self._cb_row.value()),
            "cell_mm": float(self._cb_cell.value()),
        }

    def get_computed_pixel_size(self) -> Optional[float]:
        return self._pixel_tab.get_pixel_size()

    def closeEvent(self, event) -> None:
        self._pixel_tab.cleanup()
        self._lens_tab.cleanup()
        super().closeEvent(event)

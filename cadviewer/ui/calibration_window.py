"""
CalibrationWindow — dedicated window for camera and pixel-size calibration.

Two tabs:
  1. Pixel Size Calibration: load a chessboard photo, detect corners, compute mm/px
  2. Lens Calibration: capture multiple chessboard images, run cv2.calibrateCamera

Shared chessboard parameters (cols, rows, cell size) sit above the tabs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np

from PySide6.QtCore import Qt, QSize, QUrl, QObject, QThread, Signal
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QColor, QFont, QIcon, QDesktopServices,
)
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QDoubleSpinBox, QPushButton,
    QFileDialog, QGroupBox, QTabWidget, QWidget,
    QSpinBox, QRadioButton, QButtonGroup, QListWidget,
    QListWidgetItem, QSplitter, QAbstractItemView, QComboBox,
    QProgressBar,
)

from ..core.config import AppConfig
from ..calibration.chessboard_detection import (
    detect_chessboard_corners,
    to_gray_uint8,
)
from ..camera.preview_widget import CameraPreviewWidget

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


_LENS_IMAGE_DIR = Path.home() / ".config" / "cadviewer" / "lens_calibration_images"
_LENS_MANIFEST = _LENS_IMAGE_DIR / "manifest.json"


# ── Stylesheet ──────────────────────────────────────────────────────────

_DARK_STYLE = """
    QDialog { background-color: #1e1e1e; color: #cccccc; }
    QLabel { color: #cccccc; }
    QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
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
    return to_gray_uint8(arr)

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


def _thumbnail_with_badge(pixmap: QPixmap, ok: Optional[bool]) -> QPixmap:
    """Draw a status badge on the bottom-right of a thumbnail."""
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
    if ok is True:
        painter.setPen(QColor("#66bb6a"))
        painter.drawText(size - 22, size - 6, "✓")
    elif ok is False:
        painter.setPen(QColor("#ef5350"))
        painter.drawText(size - 22, size - 6, "✗")
    else:
        painter.setPen(QColor("#f6c453"))
        painter.drawText(size - 22, size - 6, "?")
    painter.end()
    return result


# ── Collected image data ────────────────────────────────────────────────

@dataclass
class _CollectedImage:
    image: Optional[np.ndarray]
    corners: Optional[np.ndarray]
    detected: bool
    source: str
    file_path: str = ""
    detection_done: bool = False


class _CalibrationWorker(QObject):
    progress = Signal(str, int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        entries: list[tuple[int, Optional[np.ndarray], str, str]],
        cols: int,
        rows: int,
        cell_mm: float,
        flags: int,
        model: str,
    ) -> None:
        super().__init__()
        self._entries = entries
        self._cols = cols
        self._rows = rows
        self._cell_mm = cell_mm
        self._flags = flags
        self._model = model

    def run(self) -> None:
        try:
            from ..calibration.calibration_manager import CalibrationManager

            loaded: list[tuple[int, np.ndarray]] = []
            total = max(len(self._entries), 1)
            for pos, (idx, image, file_path, _source) in enumerate(self._entries, start=1):
                if image is None and file_path:
                    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
                if image is not None:
                    loaded.append((idx, image))
                self.progress.emit(
                    f"Loading calibration images {pos}/{total}",
                    int(pos * 25 / total),
                )

            if len(loaded) < 3:
                self.finished.emit({
                    "calibrated": False,
                    "message": f"Need at least 3 calibration images (have {len(loaded)}).",
                    "loaded": loaded,
                })
                return

            mgr = CalibrationManager()
            for idx, image in loaded:
                source = next(
                    (entry[3] for entry in self._entries if entry[0] == idx),
                    "",
                )
                mgr.add_image(image, source)

            def calibration_progress(stage: str, pos: int, total: int) -> None:
                total = max(int(total), 1)
                pos = max(0, min(int(pos), total))
                if stage == "Detecting chessboard corners":
                    value = 25 + int(pos * 35 / total)
                elif stage == "Solving camera calibration":
                    value = 65
                elif stage == "Building calibration report":
                    value = 70 + int(pos * 22 / total)
                else:
                    value = 60
                if pos > 0:
                    self.progress.emit(f"{stage} {pos}/{total}", value)
                else:
                    self.progress.emit(stage, value)

            self.progress.emit("Starting chessboard detection...", 25)
            result = mgr.run_calibration(
                self._cols,
                self._rows,
                self._cell_mm,
                flags=self._flags,
                model=self._model,
                progress_callback=calibration_progress,
            )
            self.progress.emit("Preparing calibration result...", 95)
            self.finished.emit({
                "calibrated": bool(result.calibrated),
                "result": result,
                "manager_images": getattr(mgr, "_images", []),
                "loaded": loaded,
            })
        except Exception as exc:
            self.error.emit(str(exc))


class _CompareModelsWorker(QObject):
    progress = Signal(str, int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        entries: list[tuple[int, Optional[np.ndarray], str, str]],
        cols: int,
        rows: int,
        cell_mm: float,
        options: list[tuple[str, str, int]],
    ) -> None:
        super().__init__()
        self._entries = entries
        self._cols = cols
        self._rows = rows
        self._cell_mm = cell_mm
        self._options = options

    def run(self) -> None:
        try:
            from ..calibration.calibration_manager import CalibrationManager

            loaded: list[tuple[int, np.ndarray]] = []
            total_entries = max(len(self._entries), 1)
            for pos, (idx, image, file_path, _source) in enumerate(self._entries, start=1):
                if image is None and file_path:
                    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
                if image is not None:
                    loaded.append((idx, image))
                self.progress.emit(
                    f"Loading calibration images {pos}/{total_entries}",
                    int(pos * 20 / total_entries),
                )

            if len(loaded) < 3:
                self.finished.emit({
                    "lines": [f"Need at least 3 calibration images (have {len(loaded)})."],
                    "loaded": loaded,
                    "manager_images": [],
                })
                return

            lines = [
                "Model comparison on current images:",
                "  Lower RMS is useful, but reject models that need poor FOV coverage "
                "or produce worse validation on measurement-area captures.",
                "",
            ]
            best_manager_images = []
            count = max(len(self._options), 1)
            for pos, (label, key, flags) in enumerate(self._options, start=1):
                self.progress.emit(
                    f"Running {label} ({pos}/{count})",
                    20 + int(pos * 75 / count),
                )
                mgr = CalibrationManager()
                for idx, image in loaded:
                    source = next(
                        (entry[3] for entry in self._entries if entry[0] == idx),
                        "",
                    )
                    mgr.add_image(image, source)
                try:
                    result = mgr.run_calibration(
                        self._cols,
                        self._rows,
                        self._cell_mm,
                        flags=flags,
                        model=key,
                    )
                except Exception as exc:
                    lines.append(f"{label}: failed ({exc})")
                    continue
                if not result.calibrated or result.dist_coeffs is None:
                    lines.append(f"{label}: failed")
                    continue
                best_manager_images = getattr(mgr, "_images", [])
                coeff_count = int(result.dist_coeffs.size)
                lines.append(
                    f"{label}: RMS {result.opencv_rms:.4f} px, "
                    f"{coeff_count} coeffs, flags {int(flags)}"
                )
            self.finished.emit({
                "lines": lines,
                "loaded": loaded,
                "manager_images": best_manager_images,
            })
        except Exception as exc:
            self.error.emit(str(exc))


class _SaveCalibrationWorker(QObject):
    progress = Signal(str, int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        entries: list[tuple[int, np.ndarray, Optional[np.ndarray]]],
        cols: int,
        rows: int,
        cell_mm: float,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> None:
        super().__init__()
        self._entries = entries
        self._cols = cols
        self._rows = rows
        self._cell_mm = cell_mm
        self._camera_matrix = camera_matrix
        self._dist_coeffs = dist_coeffs

    def run(self) -> None:
        try:
            from ..calibration.calibration_manager import CalibrationManager
            from ..calibration.coordinate_correction import CoordinateTransformer
            from ..calibration.residual_map import (
                ResidualDistortionMap, is_residual_map_safe,
            )

            undistorted_sets: list[np.ndarray] = []
            total = max(len(self._entries), 1)
            image_size = None
            for pos, (_idx, image, corners) in enumerate(self._entries, start=1):
                if image_size is None:
                    image_size = (int(image.shape[1]), int(image.shape[0]))
                undistorted = cv2.undistort(
                    image, self._camera_matrix, self._dist_coeffs,
                )
                gray = _to_gray_image(undistorted)
                found_corners, found, _method = detect_chessboard_corners(
                    gray, self._cols, self._rows,
                )
                if found:
                    undistorted_sets.append(
                        found_corners.reshape(-1, 2).astype(np.float64),
                    )
                elif corners is not None:
                    undistorted_pts = cv2.undistortPoints(
                        corners.astype(np.float32),
                        self._camera_matrix,
                        self._dist_coeffs,
                        P=self._camera_matrix,
                    )
                    undistorted_sets.append(
                        undistorted_pts.reshape(-1, 2).astype(np.float64),
                    )
                self.progress.emit(
                    f"Building correction samples {pos}/{total}",
                    int(pos * 55 / total),
                )

            residual_map_dict = {}
            coordinate_correction = {}
            correction_model_type = "none"
            if undistorted_sets and image_size is not None:
                samples = []
                corrections = []
                for pos, corners in enumerate(undistorted_sets, start=1):
                    ideal = CalibrationManager._compute_projective_ideal_grid(
                        corners, self._cols, self._rows,
                    )
                    if ideal is None:
                        continue
                    samples.append(corners)
                    corrections.append(ideal - corners)
                    self.progress.emit(
                        f"Fitting residual samples {pos}/{len(undistorted_sets)}",
                        55 + int(pos * 20 / max(len(undistorted_sets), 1)),
                    )

                residual_map = None
                if samples:
                    candidate_map = ResidualDistortionMap()
                    candidate_map.build(
                        np.vstack(samples),
                        np.vstack(corrections),
                        image_size=image_size,
                        smoothing=0.01,
                    )
                    if is_residual_map_safe(candidate_map):
                        residual_map = candidate_map
                        residual_map_dict = residual_map.to_dict()

                self.progress.emit("Building coordinate correction model...", 85)
                transformer = CoordinateTransformer()
                first_corners = undistorted_sets[0]
                if residual_map is not None and residual_map.is_built:
                    first_corners = residual_map.correct(first_corners)
                success = transformer.build_from_corners(
                    first_corners,
                    self._cols,
                    self._rows,
                    self._cell_mm,
                    "homography",
                    image_size=image_size,
                    image_count=len(undistorted_sets),
                )
                if success:
                    coordinate_correction = transformer.get_model_dict()
                    correction_model_type = "homography"

            self.finished.emit({
                "residual_map": residual_map_dict,
                "coordinate_correction": coordinate_correction,
                "correction_model_type": correction_model_type,
                "image_size": image_size,
                "undistorted_count": len(undistorted_sets),
            })
        except Exception as exc:
            self.error.emit(str(exc))


class _ImageWriteWorker(QObject):
    finished = Signal(int, bool, str)

    def __init__(self, idx: int, image: np.ndarray, file_path: str) -> None:
        super().__init__()
        self._idx = idx
        self._image = image
        self._file_path = file_path

    def run(self) -> None:
        ok = False
        message = ""
        try:
            Path(self._file_path).parent.mkdir(parents=True, exist_ok=True)
            ok = bool(cv2.imwrite(self._file_path, self._image))
            if not ok:
                message = f"Could not write {self._file_path}"
        except Exception as exc:
            message = str(exc)
        self.finished.emit(self._idx, ok, message)


# ── Pixel Size Calibration Tab ──────────────────────────────────────────

class _PixelSizeTab(QWidget):
    """Tab for computing mm/pixel from a single chessboard photo."""

    def __init__(self, parent: CalibrationWindow) -> None:
        super().__init__(parent)
        self._win = parent
        self._computed_pixel_size: Optional[float] = None
        self._captured_frame: Optional[np.ndarray] = None
        self._latest_cam_frame: Optional[np.ndarray] = None
        self._last_calibration_image: Optional[np.ndarray] = None
        self._last_corners: Optional[np.ndarray] = None
        self._last_board_center: Optional[np.ndarray] = None
        self._last_calibration_undistorted: bool = False

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
        self._btn_pose = QPushButton("Compute Mount Angles")
        self._btn_pose.clicked.connect(self._compute_mount_angles)
        self._btn_pose.setEnabled(False)
        cal_row.addWidget(self._btn_pose)
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
        """Capture a fresh full-resolution camera frame, apply undistortion."""
        frame = None
        capture_frame = getattr(self._win._camera, "capture_frame", None)
        if callable(capture_frame):
            try:
                frame = capture_frame(timeout_ms=1500)
            except TypeError:
                frame = capture_frame()
            except Exception:
                frame = None
        if frame is None and self._latest_cam_frame is not None:
            frame = self._latest_cam_frame.copy()
        if frame is None:
            self._result.setText("No frame available.")
            self._result.setStyleSheet("color: #ef5350;")
            return
        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            frame = cv2.cvtColor(frame if frame.ndim == 2 else frame[:, :, 0],
                                 cv2.COLOR_GRAY2BGR)
        corrected, applied = self._undistort(frame)
        self._last_calibration_undistorted = bool(applied)
        self._captured_frame = corrected
        self._preview.setPixmap(_numpy_to_pixmap(corrected, 400))

    def _undistort(self, frame: np.ndarray) -> tuple[np.ndarray, bool]:
        """Apply lens undistortion if calibration data exists."""
        from ..registration.auto_correspondence import undistort_if_calibrated

        return undistort_if_calibrated(frame, self._win._config)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Chessboard Image", str(Path.cwd()),
            "Images (*.png *.bmp *.tif *.tiff);;All Files (*)",
        )
        if path:
            self._path_edit.setText(path)
            self._computed_pixel_size = None
            self._captured_frame = None
            self._last_calibration_image = None
            self._last_corners = None
            self._last_board_center = None
            self._last_calibration_undistorted = False
            self._btn_pose.setEnabled(False)
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
            input_undistorted = bool(self._last_calibration_undistorted)
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
            input_undistorted = False

        cols = self._win._cb_col.value()
        rows = self._win._cb_row.value()
        cell_mm = self._win._cb_cell.value()

        gray = _to_gray_image(img)
        corners, found, method = detect_chessboard_corners(gray, cols, rows)

        if not found:
            self._result.setText(
                f"Chessboard ({cols}×{rows}) not detected in image."
            )
            self._result.setStyleSheet("color: #ef5350;")
            return

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
        self._last_calibration_image = img.copy()
        self._last_corners = corners.copy()
        self._last_board_center = np.mean(pts, axis=0)
        self._last_calibration_undistorted = input_undistorted
        self._btn_pose.setEnabled(True)
        self._win._config.pixel_size_mm = pixel_size
        self._win._config.save()

        # Draw corners on preview
        vis = img.copy()
        cv2.drawChessboardCorners(vis, (cols, rows), corners, True)
        self._draw_crosshair(vis, self._last_board_center)
        self._preview.setPixmap(_numpy_to_pixmap(vis, 400))

        self._result.setText(
            f"Detected {cols}×{rows} — {avg_px:.2f} px/cell → "
            f"{pixel_size:.4f} mm/px ({method})"
        )
        self._result.setStyleSheet("color: #66bb6a; font-weight: bold;")

    def _compute_mount_angles(self) -> None:
        if not HAS_CV2:
            self._result.setText("Error: OpenCV not available")
            self._result.setStyleSheet("color: #ef5350;")
            return
        if self._last_corners is None or self._last_calibration_image is None:
            self._result.setText("Calibrate pixel size first so chessboard corners are available.")
            self._result.setStyleSheet("color: #ef5350;")
            return

        lc = self._win._config.lens_calibration
        camera_matrix = lc.get_camera_matrix()
        dist_coeffs = lc.get_dist_coeffs()
        if camera_matrix is None or dist_coeffs is None:
            self._result.setText(
                "Run and save Lens Calibration first. Mount angles require camera intrinsics."
            )
            self._result.setStyleSheet("color: #ef5350;")
            return
        from ..registration.auto_correspondence import _scaled_camera_matrix_for_image

        camera_matrix = _scaled_camera_matrix_for_image(
            camera_matrix,
            lc,
            self._last_calibration_image.shape,
        )
        pose_dist_coeffs = (
            np.zeros_like(dist_coeffs)
            if self._last_calibration_undistorted
            else dist_coeffs
        )

        cols = self._win._cb_col.value()
        rows = self._win._cb_row.value()
        cell_mm = float(self._win._cb_cell.value())
        objp = np.zeros((cols * rows, 3), np.float32)
        objp[:, :2] = (
            np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
            * cell_mm
        )
        image_points = self._last_corners.reshape(-1, 2).astype(np.float32)

        ortho = self._estimate_orthographic_mount(
            cols, rows, cell_mm, image_points,
        )

        ok, rvec, tvec = cv2.solvePnP(
            objp, image_points, camera_matrix, pose_dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            self._result.setText("Camera pose estimation failed.")
            self._result.setStyleSheet("color: #ef5350;")
            return

        rotation, _ = cv2.Rodrigues(rvec)
        pitch, roll, yaw = self._rotation_to_pitch_roll_yaw(rotation)
        normal_cam = rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        tilt_deg = float(np.degrees(np.arccos(
            np.clip(abs(normal_cam[2]) / max(np.linalg.norm(normal_cam), 1e-12), -1.0, 1.0)
        )))
        distance_mm = float(np.linalg.norm(tvec.reshape(3)))

        vis = self._last_calibration_image.copy()
        cv2.drawChessboardCorners(vis, (cols, rows), self._last_corners, True)
        if self._last_board_center is not None:
            self._draw_crosshair(vis, self._last_board_center)
        self._draw_pose_axes(vis, camera_matrix, pose_dist_coeffs, rvec, tvec, cell_mm)
        self._preview.setPixmap(_numpy_to_pixmap(vis, 400))

        center = self._last_board_center
        center_text = (
            f"Board center: ({center[0]:.1f}, {center[1]:.1f}) px"
            if center is not None else "Board center: unavailable"
        )
        grid_tilt_text = self._format_grid_tilt_result(ortho)
        self._result.setText(
            "Pixel size and mount pose:\n"
            f"  Pixel size: {self._computed_pixel_size:.6f} mm/px\n"
            f"  Grid affine tilt: {grid_tilt_text}\n"
            f"  Grid anisotropy: {ortho['anisotropy_pct']:.3f}%\n"
            f"  Compressed board direction: {ortho['direction_deg']:+.3f} deg\n"
            f"  Affine RMS residual: {ortho['rms_px']:.4f} px\n"
            f"  solvePnP pitch: {pitch:+.3f} deg\n"
            f"  solvePnP roll:  {roll:+.3f} deg\n"
            f"  solvePnP yaw:   {yaw:+.3f} deg\n"
            f"  solvePnP optical-axis tilt: {tilt_deg:.3f} deg\n"
            f"  solvePnP translation norm: {distance_mm:.1f} mm\n"
            f"  {center_text}\n\n"
            "Grid affine tilt is the position-stable estimate for telecentric or "
            "near-orthographic metrology only when grid anisotropy is above the "
            "calibration noise floor. solvePnP is shown as a perspective-lens "
            "diagnostic and may change when the chessboard moves in the FOV."
        )
        self._result.setStyleSheet("color: #66bb6a; font-weight: bold;")

    @staticmethod
    def _format_grid_tilt_result(ortho: dict) -> str:
        anisotropy_pct = float(ortho.get("anisotropy_pct", 0.0))
        tilt_deg = float(ortho.get("tilt_deg", 0.0))
        # Below this level, a single chessboard image cannot separate real
        # mount tilt from residual lens distortion, board flatness, and corner
        # localization bias. Reporting degrees here is misleading.
        reliable_anisotropy_pct = 0.20
        reliable_tilt_floor = float(np.degrees(np.arccos(
            1.0 / (1.0 + reliable_anisotropy_pct / 100.0)
        )))
        if anisotropy_pct < reliable_anisotropy_pct:
            return (
                f"below reliable threshold "
                f"(<~{reliable_tilt_floor:.2f} deg; single-image estimate suppressed)"
            )
        return f"{tilt_deg:.3f} deg"

    @staticmethod
    def _estimate_orthographic_mount(
        cols: int,
        rows: int,
        cell_mm: float,
        image_points: np.ndarray,
    ) -> dict:
        """Estimate telecentric/orthographic plane tilt from grid anisotropy."""
        world = (
            np.mgrid[0:cols, 0:rows]
            .T.reshape(-1, 2)
            .astype(np.float64)
            * float(cell_mm)
        )
        image = image_points.reshape(-1, 2).astype(np.float64)
        design = np.column_stack([world, np.ones(len(world))])
        ax, *_ = np.linalg.lstsq(design, image[:, 0], rcond=None)
        ay, *_ = np.linalg.lstsq(design, image[:, 1], rcond=None)
        affine = np.array([
            [ax[0], ax[1], ax[2]],
            [ay[0], ay[1], ay[2]],
        ], dtype=np.float64)

        predicted = design @ np.vstack([ax, ay]).T
        residual = image - predicted
        rms_px = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))

        linear = affine[:, :2]
        _, singular_values, vt = np.linalg.svd(linear)
        s_max = float(max(singular_values))
        s_min = float(min(singular_values))
        ratio = s_min / max(s_max, 1e-12)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        tilt_deg = float(np.degrees(np.arccos(ratio)))
        anisotropy_pct = float((s_max / max(s_min, 1e-12) - 1.0) * 100.0)

        compressed_idx = int(np.argmin(singular_values))
        direction = vt[compressed_idx]
        direction_deg = float(np.degrees(np.arctan2(direction[1], direction[0])))
        if direction_deg > 90.0:
            direction_deg -= 180.0
        elif direction_deg < -90.0:
            direction_deg += 180.0

        return {
            "tilt_deg": tilt_deg,
            "anisotropy_pct": anisotropy_pct,
            "direction_deg": direction_deg,
            "rms_px": rms_px,
        }

    @staticmethod
    def _rotation_to_pitch_roll_yaw(rotation: np.ndarray) -> tuple[float, float, float]:
        """Return intrinsic XYZ-style pitch, roll, yaw from board-to-camera rotation."""
        sy = float(np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2))
        singular = sy < 1e-9
        if not singular:
            roll = np.degrees(np.arctan2(rotation[2, 1], rotation[2, 2]))
            pitch = np.degrees(np.arctan2(-rotation[2, 0], sy))
            yaw = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))
        else:
            roll = np.degrees(np.arctan2(-rotation[1, 2], rotation[1, 1]))
            pitch = np.degrees(np.arctan2(-rotation[2, 0], sy))
            yaw = 0.0
        return float(pitch), float(roll), float(yaw)

    @staticmethod
    def _draw_crosshair(image: np.ndarray, center: np.ndarray) -> None:
        x = int(round(float(center[0])))
        y = int(round(float(center[1])))
        h, w = image.shape[:2]
        length = max(25, min(w, h) // 25)
        color = (0, 255, 255)
        cv2.line(image, (max(0, x - length), y), (min(w - 1, x + length), y), color, 2, cv2.LINE_AA)
        cv2.line(image, (x, max(0, y - length)), (x, min(h - 1, y + length)), color, 2, cv2.LINE_AA)
        cv2.circle(image, (x, y), max(6, length // 5), color, 2, cv2.LINE_AA)

    @staticmethod
    def _draw_pose_axes(
        image: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        cell_mm: float,
    ) -> None:
        axis_len = float(cell_mm * 2.0)
        axes = np.array([
            [0.0, 0.0, 0.0],
            [axis_len, 0.0, 0.0],
            [0.0, axis_len, 0.0],
            [0.0, 0.0, -axis_len],
        ], dtype=np.float32)
        projected, _ = cv2.projectPoints(axes, rvec, tvec, camera_matrix, dist_coeffs)
        pts = projected.reshape(-1, 2)
        origin = tuple(np.round(pts[0]).astype(int))
        x_axis = tuple(np.round(pts[1]).astype(int))
        y_axis = tuple(np.round(pts[2]).astype(int))
        z_axis = tuple(np.round(pts[3]).astype(int))
        cv2.line(image, origin, x_axis, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.line(image, origin, y_axis, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.line(image, origin, z_axis, (255, 0, 0), 3, cv2.LINE_AA)

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
        self._calibration_model: str = "standard"
        self._calibration_flags: int = 0
        self._worker_thread: Optional[QThread] = None
        self._worker: Optional[QObject] = None
        self._image_write_threads: list[QThread] = []
        self._image_write_workers: list[QObject] = []
        self._busy = False

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
        self._btn_reload_saved = QPushButton("Reload Saved Set")
        self._btn_reload_saved.clicked.connect(self._reload_saved_set)
        btn_row.addWidget(self._btn_reload_saved)
        self._btn_open_folder = QPushButton("Open Folder")
        self._btn_open_folder.clicked.connect(self._open_saved_folder)
        btn_row.addWidget(self._btn_open_folder)
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
        cal_row.addWidget(QLabel("Model:"))
        self._model_combo = QComboBox()
        for label, key, _flags in self._calibration_model_options():
            self._model_combo.addItem(label, key)
        existing_model = getattr(
            parent._config.lens_calibration,
            "calibration_model",
            "standard",
        )
        for i in range(self._model_combo.count()):
            if self._model_combo.itemData(i) == existing_model:
                self._model_combo.setCurrentIndex(i)
                break
        cal_row.addWidget(self._model_combo)

        self._btn_run = QPushButton("Run Calibration")
        self._btn_run.setStyleSheet(
            "QPushButton { background: #264f78; color: white; font-weight: bold; }"
            "QPushButton:hover { background: #306898; }"
        )
        self._btn_run.clicked.connect(self._run_calibration)
        cal_row.addWidget(self._btn_run)
        self._btn_compare = QPushButton("Compare Models")
        self._btn_compare.clicked.connect(self._compare_models)
        cal_row.addWidget(self._btn_compare)
        cal_row.addStretch()
        layout.addLayout(cal_row)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

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
        self._load_persisted_images(silent=True)

    # ── Source toggle ────────────────────────────────────────────────

    def _on_source_changed(self, camera_selected: bool) -> None:
        if camera_selected and self._camera is not None:
            self._camera.signals.frame_ready.connect(self._preview.display_frame)
            self._btn_capture.setEnabled(not self._busy)
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
        if self._busy:
            return
        frame = None
        capture_frame = getattr(self._camera, "capture_frame", None)
        if callable(capture_frame):
            try:
                frame = capture_frame(timeout_ms=1500)
            except TypeError:
                frame = capture_frame()
            except Exception as e:
                self._status_label.setText(f"Camera capture error: {e}")
                frame = None
        if frame is None:
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
        self._add_image_entry(
            image=image,
            source=source,
            persist=True,
            detect=False,
        )

    def _add_image_entry(
        self,
        image: Optional[np.ndarray],
        source: str,
        persist: bool,
        detect: bool = False,
        detected_hint: bool = False,
        detection_done: bool = False,
        file_path: str = "",
    ) -> None:
        cols = self._win._cb_col.value()
        rows = self._win._cb_row.value()
        corners = None
        detected = bool(detected_hint)
        method = "saved" if detection_done else "pending"

        # Ensure BGR format for storage
        if image is not None and image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if detect and image is not None:
            corners, detected, method = self._detect_corners(image, cols, rows)

        entry = _CollectedImage(
            image=image,
            corners=corners,
            detected=detected,
            source=source,
            file_path=file_path,
            detection_done=detection_done,
        )
        self._collected.append(entry)

        # Thumbnail
        badge = detected if entry.detection_done else None
        if image is not None:
            pm = _numpy_to_pixmap(image, 120)
            icon_pm = _thumbnail_with_badge(pm, badge)
        else:
            icon_pm = self._placeholder_thumbnail(badge)
        item = QListWidgetItem(QIcon(icon_pm), "")
        item.setData(Qt.UserRole, len(self._collected) - 1)
        self._image_list.addItem(item)

        self._update_count()
        detail = f" ({method})" if detected else ""
        self._status_label.setText(
            f"Added: {source} — "
            f"{'corners found' if detected else 'corner detection pending'}{detail}"
        )
        if persist:
            self._persist_collected_images(write_images=False)
            if image is not None:
                self._write_entry_image_async(len(self._collected) - 1)

    @staticmethod
    def _placeholder_thumbnail(ok: Optional[bool]) -> QPixmap:
        pm = QPixmap(120, 120)
        pm.fill(QColor(30, 30, 30))
        painter = QPainter(pm)
        painter.setPen(QColor("#777"))
        painter.drawText(pm.rect(), Qt.AlignCenter, "saved")
        painter.end()
        return _thumbnail_with_badge(pm, ok)

    @staticmethod
    def _detect_corners(image: np.ndarray, cols: int, rows: int):
        """Detect chessboard corners. Returns (corners, found)."""
        if not HAS_CV2:
            return None, False, "opencv_unavailable"
        corners, found, method = detect_chessboard_corners(image, cols, rows)
        return corners, found, method

    def _remove_selected(self) -> None:
        idx = self._image_list.currentRow()
        if 0 <= idx < len(self._collected):
            self._collected.pop(idx)
            self._image_list.takeItem(idx)
            # Re-index remaining items
            for i in range(self._image_list.count()):
                self._image_list.item(i).setData(Qt.UserRole, i)
            self._update_count()
            self._persist_collected_images(write_images=False)

    def _clear_all(self) -> None:
        self._collected.clear()
        self._image_list.clear()
        self._update_count()
        self._status_label.setText("All images cleared.")
        self._clear_persisted_images()

    def _update_count(self) -> None:
        total = len(self._collected)
        checked = sum(1 for e in self._collected if e.detection_done)
        good = sum(1 for e in self._collected if e.detected)
        self._count_label.setText(
            f"Images: {total} | Checked: {checked} | Corners detected: {good}"
        )

    def _entry_image(self, entry: _CollectedImage) -> Optional[np.ndarray]:
        if entry.image is not None:
            return entry.image
        if not entry.file_path:
            return None
        image = cv2.imread(entry.file_path, cv2.IMREAD_COLOR)
        if image is not None:
            entry.image = image
        return image

    def _loaded_entries(self) -> list[_CollectedImage]:
        loaded = []
        for entry in self._collected:
            if self._entry_image(entry) is not None:
                loaded.append(entry)
        return loaded

    def _refresh_detection_state_from_manager(self, mgr) -> None:
        images = getattr(mgr, "_images", [])
        entries = [entry for entry in self._collected if entry.image is not None]
        for entry, detected in zip(entries, images):
            entry.corners = getattr(detected, "corners", None)
            entry.detected = bool(getattr(detected, "detected", False))
            entry.detection_done = True
        self._refresh_image_list_icons()
        self._update_count()
        self._persist_collected_images(write_images=False)

    def _refresh_image_list_icons(self) -> None:
        for idx, entry in enumerate(self._collected):
            if idx >= self._image_list.count():
                break
            badge = entry.detected if entry.detection_done else None
            if entry.image is not None:
                icon_pm = _thumbnail_with_badge(_numpy_to_pixmap(entry.image, 120), badge)
            else:
                icon_pm = self._placeholder_thumbnail(badge)
            self._image_list.item(idx).setIcon(QIcon(icon_pm))

    def _entry_snapshots(self) -> list[tuple[int, Optional[np.ndarray], str, str]]:
        return [
            (idx, entry.image, entry.file_path, entry.source)
            for idx, entry in enumerate(self._collected)
        ]

    def _good_entry_snapshots(
        self,
    ) -> list[tuple[int, np.ndarray, Optional[np.ndarray]]]:
        snapshots = []
        for idx, entry in enumerate(self._collected):
            if not entry.detected:
                continue
            image = self._entry_image(entry)
            if image is not None:
                snapshots.append((idx, image, entry.corners))
        return snapshots

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        can_capture = (
            not busy
            and self._camera is not None
            and self._radio_cam.isChecked()
        )
        self._btn_capture.setEnabled(can_capture)
        self._btn_add_files.setEnabled(not busy)
        self._btn_clear.setEnabled(not busy)
        self._btn_reload_saved.setEnabled(not busy)
        self._btn_remove.setEnabled(not busy)
        self._btn_run.setEnabled(not busy)
        self._btn_compare.setEnabled(not busy)
        self._btn_save.setEnabled(not busy and self._camera_matrix is not None)
        self._model_combo.setEnabled(not busy)
        self._progress.setVisible(busy)
        if not busy:
            self._progress.setValue(0)

    def _set_progress(self, message: str, value: int) -> None:
        self._status_label.setText(message)
        self._status_label.setStyleSheet("color: #ccc;")
        self._progress.setValue(max(0, min(100, int(value))))

    def _start_worker(self, worker: QObject, status: str) -> bool:
        if self._busy:
            self._status_label.setText(
                "Another calibration task is already running."
            )
            self._status_label.setStyleSheet("color: #ef5350;")
            return False
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._set_progress)
        worker.error.connect(self._worker_failed)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_worker_refs)
        self._worker_thread = thread
        self._worker = worker
        self._set_busy(True)
        self._set_progress(status, 0)
        thread.start()
        return True

    def _worker_failed(self, message: str) -> None:
        self._set_busy(False)
        self._status_label.setText(f"Calibration task failed: {message}")
        self._status_label.setStyleSheet("color: #ef5350;")

    def _clear_worker_refs(self) -> None:
        self._worker_thread = None
        self._worker = None

    def _reload_saved_set(self) -> None:
        self._load_persisted_images(silent=False)

    def _open_saved_folder(self) -> None:
        _LENS_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        url = QUrl.fromLocalFile(str(_LENS_IMAGE_DIR))
        if not QDesktopServices.openUrl(url):
            self._status_label.setText(f"Open folder failed: {_LENS_IMAGE_DIR}")
            self._status_label.setStyleSheet("color: #ef5350;")

    def _persist_collected_images(self, write_images: bool = False) -> None:
        if not HAS_CV2:
            return
        _LENS_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

        manifest = {
            "version": 1,
            "saved_at": datetime.now().isoformat(),
            "cols": int(self._win._cb_col.value()),
            "rows": int(self._win._cb_row.value()),
            "cell_mm": float(self._win._cb_cell.value()),
            "images": [],
        }
        for idx, entry in enumerate(self._collected, start=1):
            filename = self._ensure_entry_file_path(entry, idx)
            path = _LENS_IMAGE_DIR / filename
            image = entry.image
            if write_images and image is not None and not path.exists():
                if not cv2.imwrite(str(path), image):
                    continue
            if image is not None:
                h, w = image.shape[:2]
                shape = [int(w), int(h)]
            else:
                shape = []
            manifest["images"].append({
                "file": filename,
                "source": entry.source,
                "detected": bool(entry.detected),
                "detection_done": bool(entry.detection_done),
                "shape": shape,
            })
        _LENS_MANIFEST.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _ensure_entry_file_path(entry: _CollectedImage, idx: int) -> str:
        filename = Path(entry.file_path).name if entry.file_path else ""
        if not filename:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"lens_{stamp}_{idx:03d}.png"
            entry.file_path = str(_LENS_IMAGE_DIR / filename)
        return filename

    def _write_entry_image_async(self, idx: int) -> None:
        if not (0 <= idx < len(self._collected)):
            return
        entry = self._collected[idx]
        if entry.image is None:
            return
        self._ensure_entry_file_path(entry, idx + 1)
        path = Path(entry.file_path)
        if path.exists():
            return
        worker = _ImageWriteWorker(idx, entry.image.copy(), str(path))
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._image_write_finished)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(
            lambda t=thread, w=worker: self._forget_image_write_worker(t, w)
        )
        self._image_write_threads.append(thread)
        self._image_write_workers.append(worker)
        thread.start()

    def _image_write_finished(self, idx: int, ok: bool, message: str) -> None:
        if ok:
            self._persist_collected_images(write_images=False)
            return
        self._status_label.setText(f"Calibration image save failed: {message}")
        self._status_label.setStyleSheet("color: #ef5350;")

    def _forget_image_write_worker(self, thread: QThread, worker: QObject) -> None:
        try:
            self._image_write_threads.remove(thread)
        except ValueError:
            pass
        try:
            self._image_write_workers.remove(worker)
        except ValueError:
            pass

    def _clear_persisted_images(self) -> None:
        if _LENS_IMAGE_DIR.exists():
            for path in _LENS_IMAGE_DIR.glob("lens_*.png"):
                try:
                    path.unlink()
                except OSError:
                    pass
        if _LENS_MANIFEST.exists():
            try:
                _LENS_MANIFEST.unlink()
            except OSError:
                pass

    def _load_persisted_images(self, silent: bool) -> None:
        if not HAS_CV2 or not _LENS_MANIFEST.exists():
            return
        try:
            manifest = json.loads(_LENS_MANIFEST.read_text(encoding="utf-8"))
            images = manifest.get("images", [])
            if not isinstance(images, list):
                return
        except Exception as exc:
            if not silent:
                self._status_label.setText(f"Saved set load error: {exc}")
                self._status_label.setStyleSheet("color: #ef5350;")
            return

        self._collected.clear()
        self._image_list.clear()
        loaded = 0
        for item in images:
            if not isinstance(item, dict):
                continue
            filename = str(item.get("file", ""))
            if not filename:
                continue
            path = _LENS_IMAGE_DIR / filename
            if not path.exists():
                continue
            source = str(item.get("source") or filename)
            self._add_image_entry(
                image=None,
                source=source,
                persist=False,
                detect=False,
                detected_hint=bool(item.get("detected", False)),
                detection_done=bool(
                    item.get("detection_done", item.get("detected", False)),
                ),
                file_path=str(path),
            )
            loaded += 1
        self._update_count()
        if loaded and not silent:
            self._status_label.setText(
                f"Reloaded {loaded} saved calibration images from {_LENS_IMAGE_DIR}"
            )
            self._status_label.setStyleSheet("color: #66bb6a; font-weight: bold;")
        elif not loaded and not silent:
            self._status_label.setText("No saved calibration images found.")
            self._status_label.setStyleSheet("color: #ef5350;")

    # ── Calibration ──────────────────────────────────────────────────

    @staticmethod
    def _calibration_model_options() -> list[tuple[str, str, int]]:
        rational = int(getattr(cv2, "CALIB_RATIONAL_MODEL", 0)) if HAS_CV2 else 0
        thin_prism = int(getattr(cv2, "CALIB_THIN_PRISM_MODEL", 0)) if HAS_CV2 else 0
        tilted = int(getattr(cv2, "CALIB_TILTED_MODEL", 0)) if HAS_CV2 else 0
        return [
            ("Standard", "standard", 0),
            ("Rational", "rational", rational),
            (
                "Rational + Thin Prism",
                "rational_thin_prism",
                rational | thin_prism,
            ),
            (
                "Rational + Thin Prism + Tilted",
                "rational_thin_prism_tilted",
                rational | thin_prism | tilted,
            ),
        ]

    def _selected_calibration_model(self) -> tuple[str, int, str]:
        key = str(self._model_combo.currentData() or "standard")
        for label, option_key, flags in self._calibration_model_options():
            if option_key == key:
                return key, int(flags), label
        return "standard", 0, "Standard"

    def _run_calibration(self) -> None:
        if not HAS_CV2:
            self._status_label.setText("Error: OpenCV not available.")
            return

        snapshots = self._entry_snapshots()
        if len(snapshots) < 3:
            self._status_label.setText(
                f"Need at least 3 calibration images (have {len(snapshots)})."
            )
            self._status_label.setStyleSheet("color: #ef5350;")
            return

        cols = self._win._cb_col.value()
        rows = self._win._cb_row.value()
        cell_mm = self._win._cb_cell.value()

        model, flags, model_label = self._selected_calibration_model()
        worker = _CalibrationWorker(
            snapshots,
            cols,
            rows,
            cell_mm,
            flags=flags,
            model=model,
        )
        worker.finished.connect(
            lambda payload, label=model_label: self._calibration_finished(payload, label),
        )
        self._start_worker(worker, "Starting calibration...")

    def _calibration_finished(self, payload: object, model_label: str) -> None:
        self._set_busy(False)
        if not isinstance(payload, dict):
            self._status_label.setText("Calibration failed.")
            self._status_label.setStyleSheet("color: #ef5350;")
            return

        for idx, image in payload.get("loaded", []):
            if 0 <= idx < len(self._collected):
                self._collected[idx].image = image

        manager_images = payload.get("manager_images", [])
        loaded_indices = [idx for idx, _image in payload.get("loaded", [])]
        for idx, detected in zip(loaded_indices, manager_images):
            if 0 <= idx < len(self._collected):
                entry = self._collected[idx]
                entry.corners = getattr(detected, "corners", None)
                entry.detected = bool(getattr(detected, "detected", False))
                entry.detection_done = True
        self._refresh_image_list_icons()
        self._update_count()
        self._persist_collected_images(write_images=False)

        result = payload.get("result")
        if not payload.get("calibrated") or result is None:
            message = str(payload.get("message") or "Calibration failed.")
            self._status_label.setText(message)
            self._status_label.setStyleSheet("color: #ef5350;")
            return

        self._camera_matrix = result.camera_matrix
        self._dist_coeffs = result.dist_coeffs
        self._rms_error = result.opencv_rms
        self._cal_result = result
        self._calibration_model = result.calibration_model
        self._calibration_flags = result.calibration_flags

        # Display results
        mtx = result.camera_matrix
        dist = result.dist_coeffs
        fx, fy = mtx[0, 0], mtx[1, 1]
        cx, cy = mtx[0, 2], mtx[1, 2]
        lines = [
            f"Reprojection error (RMS): {result.opencv_rms:.4f} px",
            f"Model: {model_label}",
            f"OpenCV flags: {result.calibration_flags}",
            f"Images used: {result.image_count}",
            f"Corners: {result.corner_count}",
            "",
            f"Camera Matrix:",
            f"  fx = {fx:.2f}   fy = {fy:.2f}",
            f"  cx = {cx:.2f}   cy = {cy:.2f}",
            "",
            f"Distortion ({dist.size} coefficients):",
        ]
        labels = [
            "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6",
            "s1", "s2", "s3", "s4", "tauX", "tauY",
        ]
        for i, v in enumerate(dist.flatten()):
            label = labels[i] if i < len(labels) else f"d{i}"
            lines.append(f"  {label} = {v:.6f}")

        if result.report is not None:
            lines += ["", result.report.summary()]

        self._result_text.setText("\n".join(lines))
        self._btn_save.setEnabled(True)
        self._status_label.setText(
            f"Calibration complete — {model_label}, RMS: {result.opencv_rms:.4f} px "
            f"({result.image_count} images)"
        )
        self._status_label.setStyleSheet("color: #66bb6a; font-weight: bold;")

    def _compare_models(self) -> None:
        if not HAS_CV2:
            self._status_label.setText("Error: OpenCV not available.")
            return

        snapshots = self._entry_snapshots()
        if len(snapshots) < 3:
            self._status_label.setText(
                f"Need at least 3 calibration images (have {len(snapshots)})."
            )
            self._status_label.setStyleSheet("color: #ef5350;")
            return

        cols = self._win._cb_col.value()
        rows = self._win._cb_row.value()
        cell_mm = self._win._cb_cell.value()

        worker = _CompareModelsWorker(
            snapshots,
            cols,
            rows,
            cell_mm,
            self._calibration_model_options(),
        )
        worker.finished.connect(self._compare_models_finished)
        self._start_worker(worker, "Starting model comparison...")

    def _compare_models_finished(self, payload: object) -> None:
        self._set_busy(False)
        if not isinstance(payload, dict):
            self._status_label.setText("Model comparison failed.")
            self._status_label.setStyleSheet("color: #ef5350;")
            return
        for idx, image in payload.get("loaded", []):
            if 0 <= idx < len(self._collected):
                self._collected[idx].image = image

        manager_images = payload.get("manager_images", [])
        loaded_indices = [idx for idx, _image in payload.get("loaded", [])]
        for idx, detected in zip(loaded_indices, manager_images):
            if 0 <= idx < len(self._collected):
                entry = self._collected[idx]
                entry.corners = getattr(detected, "corners", None)
                entry.detected = bool(getattr(detected, "detected", False))
                entry.detection_done = True
        self._refresh_image_list_icons()
        self._update_count()
        self._persist_collected_images(write_images=False)

        lines = payload.get("lines", [])
        self._result_text.setText("\n".join(lines))
        self._status_label.setText("Model comparison complete.")
        self._status_label.setStyleSheet("color: #66bb6a; font-weight: bold;")

    def _save_to_config(self) -> None:
        if self._camera_matrix is None or self._dist_coeffs is None:
            return
        good_entries = self._good_entry_snapshots()
        if not good_entries:
            self._status_label.setText("No detected calibration images to save.")
            self._status_label.setStyleSheet("color: #ef5350;")
            return
        cols = self._win._cb_col.value()
        rows = self._win._cb_row.value()
        cell_mm = self._win._cb_cell.value()
        worker = _SaveCalibrationWorker(
            good_entries,
            cols,
            rows,
            cell_mm,
            self._camera_matrix.copy(),
            self._dist_coeffs.copy(),
        )
        worker.finished.connect(self._save_to_config_finished)
        self._start_worker(worker, "Preparing calibration save...")

    def _save_to_config_finished(self, payload: object) -> None:
        self._set_busy(False)
        try:
            self._save_to_config_impl(payload if isinstance(payload, dict) else {})
        except Exception as e:
            self._status_label.setText(f"Calibration save error: {e}")
            self._status_label.setStyleSheet("color: #ef5350;")

    def _save_to_config_impl(self, payload: dict) -> None:
        if self._camera_matrix is None:
            return
        cfg = self._win._config
        good = sum(1 for e in self._collected if e.detected)
        cfg.lens_calibration.set_from_results(
            self._camera_matrix, self._dist_coeffs,
            self._rms_error, good,
            image_size=self._calibration_image_size(),
            calibration_model=self._calibration_model,
            calibration_flags=self._calibration_flags,
        )

        cfg.lens_calibration.residual_map = payload.get("residual_map", {})
        cfg.lens_calibration.coordinate_correction = payload.get(
            "coordinate_correction", {},
        )
        cfg.lens_calibration.correction_model_type = payload.get(
            "correction_model_type", "none",
        )

        cfg.save()
        self._status_label.setText("Calibration saved to configuration.")
        self._status_label.setStyleSheet("color: #66bb6a; font-weight: bold;")

    def _calibration_image_size(self) -> tuple[int, int] | None:
        for entry in getattr(self, "_collected", []):
            image = getattr(entry, "image", None)
            if image is None:
                continue
            h, w = image.shape[:2]
            if w > 0 and h > 0:
                return int(w), int(h)
        return None


    def _undistorted_corner_sets(
        self, entries: list, cols: int, rows: int, camera_matrix, dist_coeffs,
    ) -> list[np.ndarray]:
        if camera_matrix is None or dist_coeffs is None:
            return []
        sets: list[np.ndarray] = []
        for entry in entries:
            undistorted = cv2.undistort(entry.image, camera_matrix, dist_coeffs)
            gray = _to_gray_image(undistorted)
            corners, found, _method = detect_chessboard_corners(gray, cols, rows)
            if found:
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
        self._persist_collected_images()
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
                f"({lc.image_count} images, {lc.calibration_model})"
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
